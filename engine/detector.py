import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import onnxruntime as ort

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


class EdgeDetector:
    """YOLOX-Nano ONNX Detector accelerated via AMD DirectML (DmlExecutionProvider) or Ryzen CPU."""

    def __init__(self, model_path: str = "models/yolox_nano.onnx", prefer_directml: bool = True):
        self.model_path = model_path
        self.input_size = (416, 416)
        self.strides = [8, 16, 32]
        self.grids, self.expanded_strides = self._build_grids()

        available = ort.get_available_providers()
        providers = []
        if prefer_directml and "DmlExecutionProvider" in available:
            providers.append("DmlExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
        self.active_providers = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name
        self.last_latency_ms = 0.0

    def _build_grids(self) -> Tuple[np.ndarray, np.ndarray]:
        grids = []
        expanded_strides = []
        h, w = self.input_size
        for stride in self.strides:
            gh, gw = h // stride, w // stride
            xv, yv = np.meshgrid(np.arange(gw), np.arange(gh))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))
        return np.concatenate(grids, 1), np.concatenate(expanded_strides, 1)

    def _preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        ih, iw = self.input_size
        padded = np.full((ih, iw, 3), 114, dtype=np.uint8)
        r = min(ih / img.shape[0], iw / img.shape[1])
        resized = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        )
        padded[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized
        blob = padded.transpose((2, 0, 1)).astype(np.float32)
        return np.expand_dims(blob, axis=0), r

    def detect(
        self,
        frame: np.ndarray,
        conf_thresh: float = 0.42,
        iou_thresh: float = 0.45,
        target_classes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        t0 = time.perf_counter()
        blob, ratio = self._preprocess(frame)
        outputs = self.session.run(None, {self.input_name: blob})[0]

        # Decode YOLOX outputs
        preds = outputs[0].copy()
        preds[:, :2] = (preds[:, :2] + self.grids[0]) * self.expanded_strides[0]
        preds[:, 2:4] = np.exp(preds[:, 2:4]) * self.expanded_strides[0]

        boxes_xywh = preds[:, :4]
        obj_conf = preds[:, 4:5]
        cls_scores = preds[:, 5:]
        scores = obj_conf * cls_scores

        cls_ids = np.argmax(scores, axis=1)
        max_scores = scores[np.arange(len(scores)), cls_ids]

        mask = max_scores >= conf_thresh
        if not np.any(mask):
            self.last_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return []

        boxes_xywh = boxes_xywh[mask]
        max_scores = max_scores[mask]
        cls_ids = cls_ids[mask]

        # Convert cx, cy, w, h -> x1, y1, x2, y2 and rescale to original frame
        h_orig, w_orig = frame.shape[:2]
        x1 = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0) / ratio
        y1 = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0) / ratio
        x2 = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0) / ratio
        y2 = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0) / ratio

        boxes_nms = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(boxes_nms, max_scores.tolist(), conf_thresh, iou_thresh)

        results: List[Dict[str, Any]] = []
        if len(indices) > 0:
            for i in np.array(indices).flatten():
                cid = int(cls_ids[i])
                label = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"class_{cid}"
                if target_classes and label not in target_classes:
                    continue
                bx1 = max(0, int(x1[i]))
                by1 = max(0, int(y1[i]))
                bx2 = min(w_orig - 1, int(x2[i]))
                by2 = min(h_orig - 1, int(y2[i]))
                if bx2 <= bx1 or by2 <= by1:
                    continue
                results.append(
                    {
                        "bbox": [bx1, by1, bx2, by2],
                        "label": label,
                        "score": float(max_scores[i]),
                    }
                )

        self.last_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return results

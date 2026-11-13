import time
from collections import deque
from typing import Any, Dict, List, Tuple
import math


class TrackedObject:
    """Represents a persistently tracked target across video frames."""

    def __init__(self, track_id: int, bbox: List[int], label: str, score: float):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.label = label
        self.score = score
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.missed_frames = 0
        self.history: deque = deque(maxlen=25)
        self.history.append(self.centroid)
        self.triggered_zones: set = set()

    @property
    def centroid(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    @property
    def dwell_seconds(self) -> float:
        return round(self.last_seen - self.first_seen, 1)

    def update(self, bbox: List[int], score: float, label: str) -> None:
        self.bbox = bbox
        self.score = score
        self.label = label
        self.last_seen = time.time()
        self.missed_frames = 0
        self.history.append(self.centroid)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "label": self.label,
            "score": round(float(self.score), 3),
            "centroid": list(self.centroid),
            "dwell_seconds": self.dwell_seconds,
            "trail": [list(pt) for pt in self.history],
        }


def _iou(box_a: List[int], box_b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


class MultiObjectTracker:
    """Lightweight IoU + Centroid-distance tracker optimized for real-time edge FPS."""

    def __init__(self, max_missed: int = 12, max_dist: float = 120.0):
        self.next_id = 1
        self.tracks: Dict[int, TrackedObject] = {}
        self.max_missed = max_missed
        self.max_dist = max_dist

    def update(self, detections: List[Dict[str, Any]]) -> List[TrackedObject]:
        if not detections:
            to_delete = []
            for tid, trk in self.tracks.items():
                trk.missed_frames += 1
                if trk.missed_frames > self.max_missed:
                    to_delete.append(tid)
            for tid in to_delete:
                del self.tracks[tid]
            return list(self.tracks.values())

        if not self.tracks:
            for det in detections:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = TrackedObject(tid, det["bbox"], det["label"], det["score"])
            return list(self.tracks.values())

        matched_tracks = set()
        matched_dets = set()

        track_ids = list(self.tracks.keys())
        pairs: List[Tuple[float, int, int]] = []

        for tid in track_ids:
            trk = self.tracks[tid]
            tcx, tcy = trk.centroid
            for d_idx, det in enumerate(detections):
                if det["label"] != trk.label:
                    continue
                iou_val = _iou(trk.bbox, det["bbox"])
                dx1, dy1, dx2, dy2 = det["bbox"]
                dcx, dcy = (dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0
                dist = math.hypot(dcx - tcx, dcy - tcy)

                if iou_val > 0.15 or dist < self.max_dist:
                    score = iou_val * 2.0 + max(0.0, 1.0 - dist / self.max_dist)
                    pairs.append((score, tid, d_idx))

        pairs.sort(key=lambda x: x[0], reverse=True)

        for _, tid, d_idx in pairs:
            if tid in matched_tracks or d_idx in matched_dets:
                continue
            det = detections[d_idx]
            self.tracks[tid].update(det["bbox"], det["score"], det["label"])
            matched_tracks.add(tid)
            matched_dets.add(d_idx)

        # Increment missed frames for unmatched tracks
        to_delete = []
        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid].missed_frames += 1
                if self.tracks[tid].missed_frames > self.max_missed:
                    to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        # Register new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = TrackedObject(tid, det["bbox"], det["label"], det["score"])

        return [t for t in self.tracks.values() if t.missed_frames == 0]

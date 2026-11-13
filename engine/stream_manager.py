import math
import threading
import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from engine.config import ConfigManager
from engine.detector import EdgeDetector
from engine.dvr_recorder import DVRRecorder
from engine.event_indexer import EventIndexer
from engine.tracker import MultiObjectTracker
from engine.zones import ZoneEngine


def _hex_to_bgr(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return (255, 243, 0)
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


class CameraPipeline:
    """Self-contained per-camera ingestion, AI inference, tracking, zone, and DVR thread."""

    def __init__(
        self,
        cam_cfg: Dict[str, Any],
        config_mgr: ConfigManager,
        detector: EdgeDetector,
        indexer: EventIndexer,
    ):
        self.cam_id = cam_cfg["id"]
        self.name = cam_cfg["name"]
        self.source = cam_cfg["source"]
        self.enabled = cam_cfg.get("enabled", True)

        self.config_mgr = config_mgr
        self.detector = detector
        self.indexer = indexer
        self.tracker = MultiObjectTracker(max_missed=12, max_dist=130.0)
        self.dvr = DVRRecorder(
            pre_seconds=3,
            post_seconds=4,
            fps=15,
        )

        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self.fps = 0.0
        self.latency_ms = 0.0
        self.active_tracks: List[Dict[str, Any]] = []
        self.status = "initializing"
        self._last_event_by_track: Dict[str, float] = {}

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def _generate_demo_frame(self, t: float) -> tuple:
        """Generates a crisp 960x540 simulated CCTV scene + synthetic actor coordinates."""
        w, h = 960, 540
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (16, 20, 30)

        # Architectural ground plane / driveway lines
        cv2.line(frame, (0, 260), (w, 260), (35, 45, 65), 2)
        cv2.fillPoly(
            frame,
            [np.array([[60, 540], [180, 260], [560, 260], [640, 540]])],
            (24, 30, 44),
        )
        # Subtle grid lines
        for gx in range(0, w, 80):
            cv2.line(frame, (gx, 260), (gx, h), (28, 34, 48), 1)
        for gy in range(260, h, 45):
            cv2.line(frame, (0, gy), (w, gy), (28, 34, 48), 1)

        # Building / Garage silhouette in background
        cv2.rectangle(frame, (40, 80), (500, 260), (22, 27, 38), -1)
        cv2.rectangle(frame, (120, 140), (380, 260), (30, 38, 52), -1)
        cv2.rectangle(frame, (550, 120), (920, 260), (20, 25, 36), -1)

        # Moving Actor 1: Person walking across driveway and gate
        px = int(120 + ((t * 55) % 720))
        py = int(320 + 40 * math.sin(t * 0.9))
        # Draw stylized person silhouette
        cv2.circle(frame, (px, py - 45), 14, (190, 210, 230), -1)
        cv2.rectangle(frame, (px - 16, py - 28), (px + 16, py + 24), (80, 150, 220), -1)
        cv2.rectangle(frame, (px - 14, py + 24), (px - 3, py + 62), (55, 80, 120), -1)
        cv2.rectangle(frame, (px + 3, py + 24), (px + 14, py + 62), (55, 80, 120), -1)

        # Moving Actor 2: Car entering driveway periodically
        cx = int(960 - ((t * 85) % 1080))
        cy = 390
        synthetic_dets = [
            {"bbox": [px - 22, py - 62, px + 22, py + 66], "label": "person", "score": 0.94}
        ]
        if -120 < cx < 960:
            cv2.rectangle(frame, (cx, cy), (cx + 150, cy + 68), (50, 130, 180), -1)
            cv2.rectangle(frame, (cx + 22, cy - 26), (cx + 120, cy + 2), (70, 160, 210), -1)
            cv2.circle(frame, (cx + 32, cy + 68), 16, (25, 25, 30), -1)
            cv2.circle(frame, (cx + 118, cy + 68), 16, (25, 25, 30), -1)
            synthetic_dets.append(
                {
                    "bbox": [max(0, cx - 4), cy - 30, min(w - 1, cx + 154), cy + 84],
                    "label": "car",
                    "score": 0.91,
                }
            )

        cv2.putText(
            frame,
            "SIMULATED EDGE FEED // SYNTHETIC TEST SCENE",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (100, 140, 180),
            1,
            cv2.LINE_AA,
        )
        return frame, synthetic_dets

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        src = self.source.strip()
        if src.startswith("webcam:"):
            idx = int(src.split(":", 1)[1])
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            return cap
        elif src.startswith("rtsp://") or src.startswith("http://") or src.startswith("https://"):
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        return None

    def _run_loop(self) -> None:
        is_demo = self.source.strip() == "demo:simulated"
        cap = None if is_demo else self._open_capture()
        frame_counter = 0
        fps_timer = time.perf_counter()

        while self._running:
            loop_start = time.perf_counter()
            cfg = self.config_mgr.get_all()
            features = cfg["features"]
            det_cfg = cfg["detection"]

            synthetic_dets: List[Dict[str, Any]] = []
            if is_demo:
                frame, synthetic_dets = self._generate_demo_frame(time.time())
                self.status = "online (demo)"
            else:
                if cap is None or not cap.isOpened():
                    self.status = "reconnecting"
                    time.sleep(1.5)
                    cap = self._open_capture()
                    continue
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.status = "signal_lost"
                    cap.release()
                    cap = None
                    time.sleep(1.0)
                    continue
                # Resize large RTSP streams to 960px wide for snappy iGPU processing
                if frame.shape[1] > 960:
                    scale = 960.0 / frame.shape[1]
                    frame = cv2.resize(frame, (960, int(frame.shape[0] * scale)))
                self.status = "online"

            fh, fw = frame.shape[:2]
            raw_frame = frame.copy()

            # 1. Neural Detection (DirectML / CPU)
            detections: List[Dict[str, Any]] = []
            if features.get("detection_enabled", True):
                detections = self.detector.detect(
                    frame,
                    conf_thresh=float(det_cfg.get("confidence_threshold", 0.42)),
                    iou_thresh=float(det_cfg.get("iou_threshold", 0.45)),
                    target_classes=det_cfg.get("target_classes"),
                )
                self.latency_ms = self.detector.last_latency_ms
                if is_demo and not detections:
                    # Keep real DirectML latency measured above, use synthetic actor bboxes
                    target_cls = det_cfg.get("target_classes") or []
                    detections = [
                        d for d in synthetic_dets if not target_cls or d["label"] in target_cls
                    ]

            # 2. Multi-Object Tracking
            tracks = []
            if features.get("tracking_enabled", True) and features.get("detection_enabled", True):
                tracks = self.tracker.update(detections)
                self.active_tracks = [t.to_dict() for t in tracks]
            else:
                self.active_tracks = []

            # 3. Zones & Tripwires
            zones = self.config_mgr.get_zones(self.cam_id)
            zone_alerts: List[Dict[str, Any]] = []
            if features.get("zones_enabled", True) and tracks:
                zone_alerts = ZoneEngine.evaluate(tracks, zones, fw, fh)

            # 4. Draw Overlays on Frame
            if features.get("zones_enabled", True):
                self._draw_zones(frame, zones, zone_alerts, fw, fh)

            if features.get("detection_enabled", True):
                if features.get("tracking_enabled", True):
                    self._draw_tracks(frame, tracks)
                else:
                    self._draw_raw_detections(frame, detections)

            # 5. DVR Ring Buffer & Event Logging
            if features.get("dvr_enabled", True):
                self.dvr.push_frame(frame)

            self._handle_events(frame, raw_frame, tracks, zone_alerts, features, cfg["dvr"])

            # Draw subtle HUD stamp
            ts_str = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                frame,
                f"{self.name} | {ts_str} | {self.fps:.1f} FPS | {self.latency_ms:.1f}ms",
                (16, fh - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 230, 255),
                1,
                cv2.LINE_AA,
            )

            ok_jpg, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok_jpg:
                with self._lock:
                    self._latest_jpeg = buf.tobytes()

            frame_counter += 1
            elapsed_fps = time.perf_counter() - fps_timer
            if elapsed_fps >= 1.0:
                self.fps = round(frame_counter / elapsed_fps, 1)
                frame_counter = 0
                fps_timer = time.perf_counter()

            # Cap demo loop to ~20 FPS so CPU/GPU stays cool
            if is_demo:
                dt = time.perf_counter() - loop_start
                if dt < 0.05:
                    time.sleep(0.05 - dt)

        if cap is not None:
            cap.release()

    def _handle_events(
        self,
        annotated_frame: np.ndarray,
        raw_frame: np.ndarray,
        tracks: List[Any],
        zone_alerts: List[Dict[str, Any]],
        features: Dict[str, bool],
        dvr_cfg: Dict[str, Any],
    ) -> None:
        now = time.time()
        cooldown = float(dvr_cfg.get("cooldown_seconds", 8))

        # Prioritize zone alerts
        for alert in zone_alerts:
            key = f"{alert['track_id']}_{alert['zone_id']}"
            if now - self._last_event_by_track.get(key, 0.0) >= cooldown:
                self._last_event_by_track[key] = now
                evt_id = f"evt_{int(now * 1000)}_{alert['track_id']}"
                media = (
                    self.dvr.trigger_clip(evt_id, annotated_frame, cooldown_seconds=int(cooldown))
                    if features.get("dvr_enabled", True)
                    else {"thumbnail": None, "clip": None}
                )
                self.indexer.log_event(
                    cam_id=self.cam_id,
                    cam_name=self.name,
                    label=alert["label"],
                    confidence=alert["score"],
                    track_id=alert["track_id"],
                    zone_name=alert["zone_name"],
                    zone_type=alert["zone_type"],
                    dwell_seconds=alert["dwell_seconds"],
                    thumbnail=media["thumbnail"] if media else None,
                    clip=media["clip"] if media else None,
                )

        # Log general new track appearances (when not already logged by a zone)
        for trk in tracks:
            key = f"trk_{trk.track_id}"
            if now - self._last_event_by_track.get(key, 0.0) >= max(15.0, cooldown * 2):
                self._last_event_by_track[key] = now
                evt_id = f"evt_{int(now * 1000)}_{trk.track_id}"
                media = (
                    self.dvr.trigger_clip(evt_id, annotated_frame, cooldown_seconds=int(cooldown))
                    if features.get("dvr_enabled", True)
                    else {"thumbnail": None, "clip": None}
                )
                self.indexer.log_event(
                    cam_id=self.cam_id,
                    cam_name=self.name,
                    label=trk.label,
                    confidence=trk.score,
                    track_id=trk.track_id,
                    dwell_seconds=trk.dwell_seconds,
                    thumbnail=media["thumbnail"] if media else None,
                    clip=media["clip"] if media else None,
                )

    @staticmethod
    def _draw_zones(
        frame: np.ndarray,
        zones: List[Dict[str, Any]],
        alerts: List[Dict[str, Any]],
        fw: int,
        fh: int,
    ) -> None:
        alerted_ids = {a["zone_id"] for a in alerts}
        overlay = frame.copy()

        for z in zones:
            if not z.get("enabled", True):
                continue
            pts_norm = z.get("points", [])
            if len(pts_norm) < 2:
                continue
            pts = np.array([[int(p[0] * fw), int(p[1] * fh)] for p in pts_norm], dtype=np.int32)
            is_alert = z["id"] in alerted_ids
            color = (60, 40, 255) if is_alert else _hex_to_bgr(z.get("color", "#00f3ff"))

            if z.get("type") == "polygon" and len(pts) >= 3:
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
            else:
                cv2.line(frame, tuple(pts[0]), tuple(pts[1]), color, 3, cv2.LINE_AA)

            label_txt = f"{z.get('name', 'Zone')}" + (" [ALERT]" if is_alert else "")
            cv2.putText(
                frame,
                label_txt,
                (int(pts[0][0]), max(20, int(pts[0][1]) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2 if is_alert else 1,
                cv2.LINE_AA,
            )

        cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, frame)

    @staticmethod
    def _draw_tracks(frame: np.ndarray, tracks: List[Any]) -> None:
        for trk in tracks:
            x1, y1, x2, y2 = trk.bbox
            color = (0, 243, 255) if trk.label == "person" else (180, 100, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Motion trail
            pts = list(trk.history)
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)

            tag = f"#{trk.track_id} {trk.label} {int(trk.score * 100)}% ({trk.dwell_seconds}s)"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), color, -1)
            cv2.putText(
                frame,
                tag,
                (x1 + 5, max(12, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (10, 14, 20),
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _draw_raw_detections(frame: np.ndarray, detections: List[Dict[str, Any]]) -> None:
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 243, 255), 2)
            tag = f"{det['label']} {int(det['score'] * 100)}%"
            cv2.putText(
                frame,
                tag,
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 243, 255),
                1,
                cv2.LINE_AA,
            )


class StreamManager:
    """Orchestrates all active CameraPipelines."""

    def __init__(self, config_mgr: ConfigManager, detector: EdgeDetector, indexer: EventIndexer):
        self.config_mgr = config_mgr
        self.detector = detector
        self.indexer = indexer
        self.pipelines: Dict[str, CameraPipeline] = {}
        self._lock = threading.Lock()
        self.sync_cameras()

    def sync_cameras(self) -> None:
        cams = self.config_mgr.get_cameras()
        desired_ids = {c["id"] for c in cams if c.get("enabled", True)}

        with self._lock:
            # Stop removed/disabled cameras
            for cid in list(self.pipelines.keys()):
                cam_match = next((c for c in cams if c["id"] == cid), None)
                if cid not in desired_ids or (cam_match and cam_match["source"] != self.pipelines[cid].source):
                    self.pipelines[cid].stop()
                    del self.pipelines[cid]

            # Start new cameras
            for cam in cams:
                if cam.get("enabled", True) and cam["id"] not in self.pipelines:
                    pipe = CameraPipeline(cam, self.config_mgr, self.detector, self.indexer)
                    pipe.start()
                    self.pipelines[cam["id"]] = pipe

    def get_pipeline(self, cam_id: str) -> Optional[CameraPipeline]:
        with self._lock:
            return self.pipelines.get(cam_id)

    def get_telemetry(self) -> Dict[str, Any]:
        with self._lock:
            return {
                cid: {
                    "id": p.cam_id,
                    "name": p.name,
                    "source": p.source,
                    "status": p.status,
                    "fps": p.fps,
                    "latency_ms": p.latency_ms,
                    "active_targets": len(p.active_tracks),
                }
                for cid, p in self.pipelines.items()
            }

    def stop_all(self) -> None:
        with self._lock:
            for p in self.pipelines.values():
                p.stop()
            self.pipelines.clear()

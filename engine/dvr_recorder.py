import os
import threading
import time
from collections import deque
from typing import List, Optional
import cv2
import numpy as np


class DVRRecorder:
    """Rolling ring-buffer video clip recorder & thumbnail generator."""

    def __init__(
        self,
        clips_dir: str = "storage/clips",
        thumbs_dir: str = "storage/thumbnails",
        pre_seconds: int = 3,
        post_seconds: int = 5,
        fps: int = 15,
    ):
        self.clips_dir = clips_dir
        self.thumbs_dir = thumbs_dir
        self.fps = fps
        self.pre_frames = pre_seconds * fps
        self.post_frames = post_seconds * fps

        os.makedirs(self.clips_dir, exist_ok=True)
        os.makedirs(self.thumbs_dir, exist_ok=True)

        self._lock = threading.Lock()
        self._ring_buffer: deque = deque(maxlen=max(15, self.pre_frames))
        self._active_recording: Optional[dict] = None
        self._last_record_time = 0.0

    def push_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._ring_buffer.append(frame.copy())
            if self._active_recording is not None:
                self._active_recording["frames"].append(frame.copy())
                self._active_recording["remaining"] -= 1
                if self._active_recording["remaining"] <= 0:
                    rec = self._active_recording
                    self._active_recording = None
                    threading.Thread(
                        target=self._write_clip,
                        args=(rec["path"], rec["frames"], self.fps),
                        daemon=True,
                    ).start()

    def trigger_clip(
        self, event_id: str, frame: np.ndarray, cooldown_seconds: int = 8
    ) -> Optional[dict]:
        now = time.time()
        thumb_filename = f"{event_id}.jpg"
        thumb_path = os.path.join(self.thumbs_dir, thumb_filename)
        cv2.imwrite(thumb_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        with self._lock:
            if now - self._last_record_time < cooldown_seconds or self._active_recording is not None:
                return {"thumbnail": thumb_filename, "clip": None}

            self._last_record_time = now
            clip_filename = f"{event_id}.mp4"
            clip_path = os.path.join(self.clips_dir, clip_filename)
            initial_frames = list(self._ring_buffer)
            self._active_recording = {
                "path": clip_path,
                "frames": initial_frames,
                "remaining": self.post_frames,
            }
            return {"thumbnail": thumb_filename, "clip": clip_filename}

    @staticmethod
    def _write_clip(path: str, frames: List[np.ndarray], fps: int) -> None:
        if not frames:
            return
        h, w = frames[0].shape[:2]
        # Try H.264/avc1 first for browser native playback, fallback to mp4v
        for fourcc_str in ("avc1", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
            if writer.isOpened():
                for f in frames:
                    writer.write(f)
                writer.release()
                return
            writer.release()

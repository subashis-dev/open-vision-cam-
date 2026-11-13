import json
import os
import threading
from typing import Any, Dict, List

DEFAULT_CONFIG: Dict[str, Any] = {
    "features": {
        "detection_enabled": True,
        "tracking_enabled": True,
        "zones_enabled": True,
        "dvr_enabled": True,
        "nlp_search_enabled": True,
        "ai_summary_enabled": True,
        "sound_alerts_enabled": False,
    },
    "detection": {
        "confidence_threshold": 0.42,
        "iou_threshold": 0.45,
        "target_classes": [
            "person",
            "car",
            "truck",
            "bus",
            "motorcycle",
            "bicycle",
            "dog",
            "cat",
            "backpack",
            "handbag",
            "suitcase",
        ],
        "prefer_directml": True,
    },
    "dvr": {
        "pre_event_seconds": 3,
        "post_event_seconds": 5,
        "cooldown_seconds": 8,
        "fps": 15,
    },
    "cameras": [
        {
            "id": "cam-1",
            "name": "Front Yard (Demo Feed)",
            "source": "demo:simulated",
            "enabled": True,
        }
    ],
    "zones": {
        "cam-1": [
            {
                "id": "zone-driveway",
                "name": "Driveway Perimeter",
                "type": "polygon",
                "color": "#00f3ff",
                "points": [[0.10, 0.52], [0.58, 0.52], [0.65, 0.94], [0.05, 0.94]],
                "enabled": True,
            },
            {
                "id": "line-gate",
                "name": "Main Gate Tripwire",
                "type": "tripwire",
                "color": "#ff007f",
                "points": [[0.60, 0.45], [0.95, 0.72]],
                "enabled": True,
            },
        ]
    },
}


class ConfigManager:
    """Thread-safe configuration & modular feature-toggle manager."""

    def __init__(self, path: str = "storage/config.json"):
        self.path = path
        self._lock = threading.RLock()
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                # Merge missing keys from DEFAULT_CONFIG
                merged = json.loads(json.dumps(DEFAULT_CONFIG))
                for k, v in loaded.items():
                    if isinstance(v, dict) and isinstance(merged.get(k), dict):
                        merged[k].update(v)
                    else:
                        merged[k] = v
                return merged
            except Exception:
                pass
        return json.loads(json.dumps(DEFAULT_CONFIG))

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with self._lock:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)

    def get_all(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self.data))

    def update_features(self, updates: Dict[str, bool]) -> Dict[str, Any]:
        with self._lock:
            for k, v in updates.items():
                if k in self.data["features"]:
                    self.data["features"][k] = bool(v)
            self.save()
            return self.get_all()

    def update_detection(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            for k, v in updates.items():
                if k in self.data["detection"]:
                    self.data["detection"][k] = v
            self.save()
            return self.get_all()

    def get_zones(self, cam_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return json.loads(json.dumps(self.data.get("zones", {}).get(cam_id, [])))

    def set_zones(self, cam_id: str, zones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        with self._lock:
            if "zones" not in self.data:
                self.data["zones"] = {}
            self.data["zones"][cam_id] = zones
            self.save()
            return self.get_zones(cam_id)

    def get_cameras(self) -> List[Dict[str, Any]]:
        with self._lock:
            return json.loads(json.dumps(self.data.get("cameras", [])))

    def upsert_camera(self, cam: Dict[str, Any]) -> List[Dict[str, Any]]:
        with self._lock:
            cams = self.data.get("cameras", [])
            for idx, existing in enumerate(cams):
                if existing["id"] == cam["id"]:
                    cams[idx] = cam
                    break
            else:
                cams.append(cam)
            self.data["cameras"] = cams
            self.save()
            return self.get_cameras()

    def delete_camera(self, cam_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            self.data["cameras"] = [c for c in self.data.get("cameras", []) if c["id"] != cam_id]
            self.save()
            return self.get_cameras()

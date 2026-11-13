import json
import os
import re
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


class EventIndexer:
    """Thread-safe semantic event store with natural language search indexing."""

    def __init__(self, path: str = "storage/events.json", max_events: int = 1000):
        self.path = path
        self.max_events = max_events
        self._lock = threading.RLock()
        self.events: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
            except Exception:
                pass
        return []

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.events[: self.max_events], f, indent=2)

    def log_event(
        self,
        cam_id: str,
        cam_name: str,
        label: str,
        confidence: float,
        track_id: int,
        zone_name: Optional[str] = None,
        zone_type: Optional[str] = None,
        dwell_seconds: float = 0.0,
        thumbnail: Optional[str] = None,
        clip: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = time.time()
        dt = datetime.fromtimestamp(now)
        event_id = f"evt_{int(now * 1000)}_{track_id}"

        if zone_name:
            verb = "crossed" if zone_type == "tripwire" else "entered"
            summary = f"{label.capitalize()} (#{track_id}) {verb} {zone_name}"
            severity = "alert"
        else:
            summary = f"{label.capitalize()} (#{track_id}) detected on {cam_name}"
            severity = "info"

        # Rich searchable semantic tags
        keywords = [
            label.lower(),
            cam_name.lower(),
            cam_id.lower(),
            severity,
            dt.strftime("%I:%M %p").lower(),
            dt.strftime("%H:%M").lower(),
            dt.strftime("%Y-%m-%d"),
            summary.lower(),
        ]
        if zone_name:
            keywords.append(zone_name.lower())
        if zone_type:
            keywords.append(zone_type.lower())

        event = {
            "id": event_id,
            "timestamp": now,
            "time_str": dt.strftime("%I:%M:%S %p"),
            "date_str": dt.strftime("%Y-%m-%d"),
            "camera_id": cam_id,
            "camera_name": cam_name,
            "label": label,
            "confidence": round(confidence, 3),
            "track_id": track_id,
            "zone_name": zone_name,
            "zone_type": zone_type,
            "dwell_seconds": dwell_seconds,
            "severity": severity,
            "summary": summary,
            "thumbnail": thumbnail,
            "clip": clip,
            "search_blob": " ".join(keywords),
        }

        with self._lock:
            self.events.insert(0, event)
            if len(self.events) > self.max_events:
                self.events = self.events[: self.max_events]
            self._save()
        return event

    def search(
        self,
        query: str = "",
        camera_id: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            results = list(self.events)

        if camera_id and camera_id != "all":
            results = [e for e in results if e.get("camera_id") == camera_id]

        if severity and severity != "all":
            results = [e for e in results if e.get("severity") == severity]

        q = query.strip().lower()
        if q:
            # Handle natural language time hints
            now = time.time()
            if "last hour" in q or "recent" in q:
                results = [e for e in results if now - e.get("timestamp", 0) <= 3600]
                q = q.replace("last hour", "").replace("recent", "").strip()
            elif "today" in q:
                today = datetime.now().strftime("%Y-%m-%d")
                results = [e for e in results if e.get("date_str") == today]
                q = q.replace("today", "").strip()

            # Remove stop words for natural language queries like "person in driveway"
            stop_words = {"in", "at", "on", "the", "a", "an", "with", "near", "show", "me", "find"}
            tokens = [t for t in re.split(r"\s+", q) if t and t not in stop_words]

            # Map synonyms (e.g. human -> person, vehicle -> car/truck/bus)
            synonyms = {
                "human": ["person"],
                "people": ["person"],
                "man": ["person"],
                "woman": ["person"],
                "intruder": ["person", "alert"],
                "vehicle": ["car", "truck", "bus", "motorcycle"],
                "pet": ["dog", "cat"],
                "animal": ["dog", "cat", "bird"],
                "package": ["backpack", "handbag", "suitcase"],
                "alarm": ["alert", "tripwire", "polygon"],
            }

            if tokens:
                filtered = []
                for e in results:
                    blob = e.get("search_blob", "")
                    match_all = True
                    for tok in tokens:
                        expanded = synonyms.get(tok, [tok])
                        if not any(exp in blob for exp in expanded):
                            match_all = False
                            break
                    if match_all:
                        filtered.append(e)
                results = filtered

        return results[:limit]

    def clear(self) -> None:
        with self._lock:
            self.events = []
            self._save()

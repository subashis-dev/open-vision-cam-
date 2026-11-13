from collections import Counter
from datetime import datetime
from typing import Any, Dict, List


class IncidentSummarizer:
    """Generates natural-language AI security story briefings from indexed events."""

    @staticmethod
    def generate_briefing(events: List[Dict[str, Any]], hours: int = 24) -> Dict[str, Any]:
        if not events:
            return {
                "generated_at": datetime.now().strftime("%I:%M:%S %p"),
                "total_events": 0,
                "alerts_count": 0,
                "headline": "Perimeter Clear — No Incidents Logged",
                "narrative": (
                    "OpenVisionCam has been monitoring your feeds with zero security intrusions "
                    "or object triggers recorded in the current session."
                ),
                "highlights": [],
                "breakdown": {},
            }

        total = len(events)
        alerts = [e for e in events if e.get("severity") == "alert"]
        labels = Counter(e.get("label", "object") for e in events)
        zones = Counter(e.get("zone_name") for e in events if e.get("zone_name"))

        top_obj, top_count = labels.most_common(1)[0]
        latest = events[0]

        if alerts:
            headline = f"{len(alerts)} Perimeter Alert(s) — Latest: {alerts[0]['summary']}"
        else:
            headline = f"{total} Activity Event(s) Logged — Primary Target: {top_obj.capitalize()}"

        parts = [
            f"During the active monitoring window, OpenVisionCam logged {total} vision event(s) "
            f"({len(alerts)} perimeter intrusion alert(s)).",
            f"The most frequently observed target was '{top_obj}' ({top_count} occurrence(s)).",
        ]

        if zones:
            top_zone, z_count = zones.most_common(1)[0]
            parts.append(
                f"The highest activity zone was '{top_zone}' with {z_count} boundary trigger(s)."
            )

        parts.append(
            f"Most recent activity occurred at {latest.get('time_str')} on "
            f"{latest.get('camera_name')}: {latest.get('summary')} "
            f"(confidence {int(latest.get('confidence', 0) * 100)}%)."
        )

        highlights = []
        for e in (alerts[:5] if alerts else events[:5]):
            highlights.append(
                f"[{e.get('time_str')}] {e.get('summary')} "
                f"({int(e.get('confidence', 0) * 100)}% conf)"
            )

        return {
            "generated_at": datetime.now().strftime("%I:%M:%S %p"),
            "total_events": total,
            "alerts_count": len(alerts),
            "headline": headline,
            "narrative": " ".join(parts),
            "highlights": highlights,
            "breakdown": dict(labels),
        }

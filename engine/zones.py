from typing import Any, Dict, List, Tuple
import cv2
import numpy as np


def _point_in_polygon(pt: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
    if len(polygon) < 3:
        return False
    poly_np = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(poly_np, (float(pt[0]), float(pt[1])), False) >= 0


def _ccw(A: Tuple[int, int], B: Tuple[int, int], C: Tuple[int, int]) -> bool:
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def _segments_intersect(
    A: Tuple[int, int], B: Tuple[int, int], C: Tuple[int, int], D: Tuple[int, int]
) -> bool:
    return _ccw(A, C, D) != _ccw(B, C, D) and _ccw(A, B, C) != _ccw(A, B, D)


class ZoneEngine:
    """Evaluates virtual intrusion polygons and perimeter tripwire crossings."""

    @staticmethod
    def evaluate(
        tracks: List[Any],
        zones: List[Dict[str, Any]],
        frame_w: int,
        frame_h: int,
    ) -> List[Dict[str, Any]]:
        alerts: List[Dict[str, Any]] = []
        if not zones or not tracks:
            return alerts

        for zone in zones:
            if not zone.get("enabled", True):
                continue
            pts_norm = zone.get("points", [])
            if len(pts_norm) < 2:
                continue

            pts_px = [(int(p[0] * frame_w), int(p[1] * frame_h)) for p in pts_norm]
            z_type = zone.get("type", "polygon")
            z_id = zone.get("id", "zone")
            z_name = zone.get("name", "Zone")

            for trk in tracks:
                # Use bottom-center (feet/wheels) for accurate ground-plane zone checks
                x1, y1, x2, y2 = trk.bbox
                foot_pt = (int((x1 + x2) / 2), int(y2))

                if z_type == "polygon" and len(pts_px) >= 3:
                    if _point_in_polygon(foot_pt, pts_px) or _point_in_polygon(trk.centroid, pts_px):
                        alerts.append(
                            {
                                "zone_id": z_id,
                                "zone_name": z_name,
                                "zone_type": "polygon",
                                "track_id": trk.track_id,
                                "label": trk.label,
                                "score": trk.score,
                                "dwell_seconds": trk.dwell_seconds,
                            }
                        )
                elif z_type == "tripwire" and len(pts_px) >= 2:
                    if len(trk.history) >= 2:
                        prev_pt = trk.history[-2]
                        curr_pt = trk.history[-1]
                        if _segments_intersect(prev_pt, curr_pt, pts_px[0], pts_px[1]):
                            alerts.append(
                                {
                                    "zone_id": z_id,
                                    "zone_name": z_name,
                                    "zone_type": "tripwire",
                                    "track_id": trk.track_id,
                                    "label": trk.label,
                                    "score": trk.score,
                                    "dwell_seconds": trk.dwell_seconds,
                                }
                            )
        return alerts

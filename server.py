import asyncio
import os
import platform
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import psutil
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine.config import ConfigManager
from engine.detector import EdgeDetector
from engine.event_indexer import EventIndexer
from engine.stream_manager import StreamManager
from engine.summarizer import IncidentSummarizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

config_mgr = ConfigManager("storage/config.json")
indexer = EventIndexer("storage/events.json")
detector = EdgeDetector(
    model_path="models/yolox_nano.onnx",
    prefer_directml=config_mgr.get_all()["detection"].get("prefer_directml", True),
)
stream_mgr: Optional[StreamManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global stream_mgr
    stream_mgr = StreamManager(config_mgr, detector, indexer)
    yield
    if stream_mgr:
        stream_mgr.stop_all()


app = FastAPI(title="OpenVisionCam Edge NVR", version="1.0.0", lifespan=lifespan)

# Mount web static assets
app.mount("/static", StaticFiles(directory="web"), name="static")


class FeatureTogglePayload(BaseModel):
    detection_enabled: Optional[bool] = None
    tracking_enabled: Optional[bool] = None
    zones_enabled: Optional[bool] = None
    dvr_enabled: Optional[bool] = None
    nlp_search_enabled: Optional[bool] = None
    ai_summary_enabled: Optional[bool] = None
    sound_alerts_enabled: Optional[bool] = None


class DetectionSettingsPayload(BaseModel):
    confidence_threshold: Optional[float] = None
    target_classes: Optional[List[str]] = None


class CameraPayload(BaseModel):
    id: str
    name: str
    source: str
    enabled: bool = True


class ZoneItem(BaseModel):
    id: str
    name: str
    type: str  # "polygon" or "tripwire"
    color: str = "#00f3ff"
    points: List[List[float]]
    enabled: bool = True


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    with open("web/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/system")
async def get_system_status():
    mem = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_percent": mem.percent,
        "ram_used_gb": round(mem.used / (1024 ** 3), 2),
        "ram_total_gb": round(mem.total / (1024 ** 3), 2),
        "processor": platform.processor() or "AMD Ryzen Edge CPU",
        "providers": detector.active_providers,
        "directml_active": "DmlExecutionProvider" in detector.active_providers,
        "last_latency_ms": detector.last_latency_ms,
        "cameras": stream_mgr.get_telemetry() if stream_mgr else {},
    }


@app.get("/api/config")
async def get_config():
    return config_mgr.get_all()


@app.post("/api/config/features")
async def update_features(payload: FeatureTogglePayload):
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    return config_mgr.update_features(updates)


@app.post("/api/config/detection")
async def update_detection(payload: DetectionSettingsPayload):
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    return config_mgr.update_detection(updates)


@app.get("/api/cameras")
async def list_cameras():
    return {"cameras": config_mgr.get_cameras()}


@app.post("/api/cameras")
async def upsert_camera(payload: CameraPayload):
    cams = config_mgr.upsert_camera(payload.model_dump())
    if stream_mgr:
        stream_mgr.sync_cameras()
    return {"cameras": cams}


@app.delete("/api/cameras/{cam_id}")
async def delete_camera(cam_id: str):
    cams = config_mgr.delete_camera(cam_id)
    if stream_mgr:
        stream_mgr.sync_cameras()
    return {"cameras": cams}


@app.get("/api/zones/{cam_id}")
async def get_zones(cam_id: str):
    return {"zones": config_mgr.get_zones(cam_id)}


@app.post("/api/zones/{cam_id}")
async def save_zones(cam_id: str, zones: List[ZoneItem]):
    saved = config_mgr.set_zones(cam_id, [z.model_dump() for z in zones])
    return {"zones": saved}


@app.get("/api/events")
async def get_events(
    q: str = Query(default=""),
    cam_id: str = Query(default="all"),
    severity: str = Query(default="all"),
    limit: int = Query(default=100, le=500),
):
    cfg = config_mgr.get_all()
    if not cfg["features"].get("nlp_search_enabled", True) and q.strip():
        return {"events": [], "disabled": True}
    events = indexer.search(query=q, camera_id=cam_id, severity=severity, limit=limit)
    return {"events": events, "total": len(events)}


@app.delete("/api/events")
async def clear_events():
    indexer.clear()
    return {"status": "cleared"}


@app.get("/api/summary")
async def get_summary():
    cfg = config_mgr.get_all()
    if not cfg["features"].get("ai_summary_enabled", True):
        return {"disabled": True}
    events = indexer.search(limit=250)
    return IncidentSummarizer.generate_briefing(events)


@app.get("/api/clips/{filename}")
async def get_clip(filename: str):
    safe_name = os.path.basename(filename)
    path = os.path.join("storage", "clips", safe_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Clip not found or still encoding")
    return FileResponse(path, media_type="video/mp4", filename=safe_name)


@app.get("/api/thumbnails/{filename}")
async def get_thumbnail(filename: str):
    safe_name = os.path.basename(filename)
    path = os.path.join("storage", "thumbnails", safe_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(path, media_type="image/jpeg")


async def _mjpeg_generator(cam_id: str):
    while True:
        pipe = stream_mgr.get_pipeline(cam_id) if stream_mgr else None
        if pipe:
            jpg = pipe.get_jpeg()
            if jpg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )
        await asyncio.sleep(0.04)


@app.get("/api/stream/{cam_id}")
async def video_stream(cam_id: str):
    return StreamingResponse(
        _mjpeg_generator(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/api/snapshot/{cam_id}")
async def capture_snapshot(cam_id: str):
    pipe = stream_mgr.get_pipeline(cam_id) if stream_mgr else None
    if not pipe:
        raise HTTPException(status_code=404, detail="Camera not found")
    jpg = pipe.get_jpeg()
    if not jpg:
        raise HTTPException(status_code=503, detail="Stream frame not ready")

    import time as _t
    evt_id = f"snap_{int(_t.time() * 1000)}"
    thumb_name = f"{evt_id}.jpg"
    thumb_path = os.path.join("storage", "thumbnails", thumb_name)
    os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
    with open(thumb_path, "wb") as f:
        f.write(jpg)

    event = indexer.log_event(
        cam_id=cam_id,
        cam_name=pipe.name,
        label="snapshot",
        confidence=1.0,
        track_id=0,
        zone_name="Manual Capture",
        zone_type="snapshot",
        thumbnail=thumb_name,
    )
    return {"status": "captured", "event": event}


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 66)
    print("  👁️  OPENVISIONCAM — SOVEREIGN EDGE AI NVR")
    print(f"  ⚡ Execution Providers : {detector.active_providers}")
    print("  🌐 Web Dashboard       : http://localhost:8085")
    print("=" * 66 + "\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8085, reload=False)

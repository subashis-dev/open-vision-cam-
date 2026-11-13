# 👁️ OpenVisionCam

[![License: MIT](https://img.shields.io/badge/License-MIT-00f3ff.svg)](LICENSE)
[![Engine](https://img.shields.io/badge/Inference-ONNX_Runtime_DirectML-ff007f.svg)](#)
[![Hardware](https://img.shields.io/badge/Optimized_For-AMD_Ryzen_%2B_Radeon_iGPU-10b981.svg)](#)
[![Privacy](https://img.shields.io/badge/Privacy-100%25_Local_Edge_NVR-8b5cf6.svg)](#)

**OpenVisionCam** is a 100% local, sovereign AI Network Video Recorder (NVR) and smart surveillance dashboard engineered from the ground up to run smoothly on **AMD Ryzen CPUs and Radeon Integrated GPUs** via **Microsoft DirectML**.

No cloud subscriptions. No proprietary camera lock-in. Every feature in the dashboard is completely modular and toggleable ON/OFF in real time.

---

## ✨ Modular Feature Suite (All Toggleable)

| Feature | Description | Control |
| :--- | :--- | :---: |
| **AMD DirectML Neural Detector** | Runs **YOLOX-Nano** (`416x416` ONNX) on your integrated Radeon GPU (`DmlExecutionProvider`) with automatic multi-core Ryzen CPU fallback. | `ON / OFF` |
| **Multi-Target Centroid Tracker** | Assigns persistent tracking IDs (`#1 person`, `#2 car`), renders motion trajectory trails, and calculates dwell time. | `ON / OFF` |
| **Interactive Zones & Tripwires** | Draw custom polygon intrusion zones and virtual perimeter tripwire lines directly onto the live video canvas. | `ON / OFF` |
| **Rolling-Buffer Event DVR** | Continuously buffers pre-event frames and automatically writes MP4 clips & JPG snapshots when an intrusion occurs. | `ON / OFF` |
| **Natural Language Event Search** | Query incidents in plain English (e.g., *"person in driveway"*, *"vehicle"*, *"tripwire"*, *"last hour"*). | `ON / OFF` |
| **AI Security Briefings** | Generates automated natural-language incident story summaries and activity highlights. | `ON / OFF` |
| **Multi-Source Ingestion** | Connect RTSP IP cameras (`rtsp://...`), local webcams (`webcam:0`), or test immediately with the built-in simulated feed. | `Dynamic` |

---

## 🏗️ Project Architecture

```text
open-vision-cam/
├── engine/
│   ├── config.py             # Thread-safe modular feature toggle & zone store
│   ├── detector.py           # YOLOX-Nano ONNX engine (DirectML iGPU + CPU)
│   ├── tracker.py            # IoU + Centroid multi-object tracker & trails
│   ├── zones.py              # Polygon intrusion & tripwire crossing evaluator
│   ├── dvr_recorder.py       # Ring-buffer MP4 clip & thumbnail recorder
│   ├── event_indexer.py      # Semantic event store & NLP query matcher
│   ├── summarizer.py         # Automated security incident story generator
│   └── stream_manager.py     # Resilient RTSP, Webcam & Simulated Demo pipeline
├── web/
│   ├── index.html            # Cyberpunk Glassmorphic NVR Dashboard
│   ├── css/style.css         # Custom HUD & toggle switch styling
│   └── js/app.js             # Live telemetry, canvas drawing & DVR playback
├── models/
│   └── yolox_nano.onnx       # Lightweight 3.6 MB edge detection model (80 classes)
├── storage/                  # Local event logs, thumbnails & MP4 clips
└── server.py                 # FastAPI backend & low-latency MJPEG streamer
```

---

## 🚀 Quick Start

### 1. Clone & Enter Repository
```bash
git clone https://github.com/subashis-dev/open-vision-cam.git
cd open-vision-cam
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch OpenVisionCam
```bash
python server.py
```

Open your browser and navigate to:
```text
http://localhost:8085
```

---

## 📷 Connecting Your Own Cameras

Click **"+ Add / Switch Camera"** in the top-right HUD and enter any standard source:
* **RTSP IP Camera (Wyze, Tapo, Reolink, Hikvision, Dahua):**
  `rtsp://username:password@192.168.1.50:554/stream1`
* **Local USB / Laptop Webcam:**
  `webcam:0`
* **Built-in Demo Scene (No Hardware Needed):**
  `demo:simulated`

---

## 📜 License

This project is licensed under the [MIT License](LICENSE) — Copyright (c) 2026 Subashis Roy Bhowmik.

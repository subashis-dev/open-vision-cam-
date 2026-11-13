let activeCamId = "cam-1";
let allCameras = [];
let isGridView = false;
let currentZones = [];
let drawingMode = null; // "polygon" | "tripwire" | null
let drawingPoints = [];
let searchTimeout = null;
let lastKnownAlertId = null;
let soundAlertsEnabled = false;
let activeTargetGroups = new Set(["person", "vehicle", "bicycle", "pet", "package"]);

const TARGET_GROUP_MAP = {
  person: ["person"],
  vehicle: ["car", "truck", "bus", "motorcycle"],
  bicycle: ["bicycle"],
  pet: ["dog", "cat"],
  package: ["backpack", "handbag", "suitcase"],
};

const canvas = document.getElementById("zone-canvas");
const ctx = canvas.getContext("2d");

async function init() {
  await loadConfig();
  await loadCameras();
  await loadZones();
  await loadEvents();
  await loadSummary();
  await pollSystem();

  setInterval(pollSystem, 2000);
  setInterval(loadEvents, 4000);
  setInterval(loadSummary, 8000);
}

async function loadConfig() {
  try {
    const res = await fetch("/api/config");
    const cfg = await res.json();
    const f = cfg.features || {};

    document.getElementById("tog-detection").checked = !!f.detection_enabled;
    document.getElementById("tog-tracking").checked = !!f.tracking_enabled;
    document.getElementById("tog-zones").checked = !!f.zones_enabled;
    document.getElementById("tog-dvr").checked = !!f.dvr_enabled;
    document.getElementById("tog-nlp").checked = !!f.nlp_search_enabled;
    document.getElementById("tog-summary").checked = !!f.ai_summary_enabled;
    document.getElementById("tog-sound").checked = !!f.sound_alerts_enabled;
    soundAlertsEnabled = !!f.sound_alerts_enabled;

    const confPct = Math.round((cfg.detection?.confidence_threshold ?? 0.42) * 100);
    document.getElementById("conf-slider").value = confPct;
    document.getElementById("conf-label").innerText = `${confPct}%`;
  } catch (err) {
    console.error("Failed to load config:", err);
  }
}

async function toggleFeature(key, enabled) {
  if (key === "sound_alerts_enabled") {
    soundAlertsEnabled = enabled;
  }
  await fetch("/api/config/features", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ [key]: enabled }),
  });
  if (key === "ai_summary_enabled") {
    loadSummary();
  }
  if (key === "nlp_search_enabled") {
    loadEvents();
  }
}

function updateConfLabel(val) {
  document.getElementById("conf-label").innerText = `${val}%`;
}

async function saveConfidence(val) {
  await fetch("/api/config/detection", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confidence_threshold: parseInt(val, 10) / 100.0 }),
  });
}

async function loadCameras() {
  const res = await fetch("/api/cameras");
  const data = await res.json();
  allCameras = data.cameras || [];
  const select = document.getElementById("camera-select");
  select.innerHTML = "";
  allCameras.forEach((cam) => {
    const opt = document.createElement("option");
    opt.value = cam.id;
    opt.textContent = `${cam.name} (${cam.source})`;
    if (cam.id === activeCamId) opt.selected = true;
    select.appendChild(opt);
  });
  if (isGridView) renderGridView();
}

function toggleGridView() {
  isGridView = !isGridView;
  const singleEl = document.getElementById("viewport-wrapper");
  const gridEl = document.getElementById("grid-viewport");
  const btn = document.getElementById("btn-grid-toggle");

  if (isGridView) {
    singleEl.classList.add("hidden");
    gridEl.classList.remove("hidden");
    btn.classList.add("bg-violet-500/30", "border-violet-400");
    btn.innerText = "▣ Single View";
    renderGridView();
  } else {
    gridEl.classList.add("hidden");
    singleEl.classList.remove("hidden");
    btn.classList.remove("bg-violet-500/30", "border-violet-400");
    btn.innerText = "⊞ Grid View";
  }
}

function renderGridView() {
  const gridEl = document.getElementById("grid-viewport");
  gridEl.innerHTML = allCameras
    .map(
      (cam) => `
      <div onclick="switchCamera('${cam.id}'); toggleGridView();" class="cursor-pointer relative bg-black rounded-xl overflow-hidden border border-white/10 hover:border-cyan-400/60 transition aspect-video group">
        <img src="/api/stream/${cam.id}?t=${Date.now()}" class="w-full h-full object-contain" />
        <div class="absolute top-2 left-2 px-2.5 py-1 rounded-lg bg-slate-950/80 border border-white/10 text-[11px] font-mono text-cyan-300">
          ${cam.name}
        </div>
        <div class="hidden group-hover:flex absolute inset-0 bg-cyan-500/10 items-center justify-center text-xs font-mono text-white">
          Click to Focus
        </div>
      </div>`
    )
    .join("");
}

async function takeSnapshot() {
  try {
    const res = await fetch(`/api/snapshot/${activeCamId}`, { method: "POST" });
    if (res.ok) {
      await loadEvents();
      await loadSummary();
    }
  } catch (err) {
    console.error("Snapshot error:", err);
  }
}

async function toggleTargetGroup(group, btnEl) {
  if (activeTargetGroups.has(group)) {
    activeTargetGroups.delete(group);
    btnEl.className = "target-pill px-2.5 py-1 rounded-lg bg-slate-900 border border-white/10 text-slate-500 transition";
  } else {
    activeTargetGroups.add(group);
    btnEl.className = "target-pill px-2.5 py-1 rounded-lg bg-cyan-500/20 border border-cyan-400/50 text-cyan-200 transition";
  }

  const classes = [];
  activeTargetGroups.forEach((g) => {
    (TARGET_GROUP_MAP[g] || []).forEach((cls) => classes.push(cls));
  });

  await fetch("/api/config/detection", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target_classes: classes }),
  });
}

async function exportReport(fmt) {
  const [sumRes, evtRes] = await Promise.all([
    fetch("/api/summary"),
    fetch("/api/events?limit=250"),
  ]);
  const summary = await sumRes.json();
  const eventsData = await evtRes.json();
  const dateTag = new Date().toISOString().slice(0, 10);

  let content = "";
  let mime = "text/plain";
  let ext = "txt";

  if (fmt === "json") {
    content = JSON.stringify({ generated_at: new Date().toISOString(), summary, events: eventsData.events }, null, 2);
    mime = "application/json";
    ext = "json";
  } else {
    const lines = [
      "============================================================",
      "  OPENVISIONCAM — AI SECURITY INCIDENT DOSSIER",
      `  Generated: ${new Date().toLocaleString()}`,
      "============================================================",
      "",
      `HEADLINE : ${summary.headline || "N/A"}`,
      `SUMMARY  : ${summary.narrative || "N/A"}`,
      "",
      "--- TOP HIGHLIGHTS ---",
      ...(summary.highlights || ["No alerts recorded."]),
      "",
      "--- RECENT EVENT LOG ---",
      ...(eventsData.events || []).map(
        (e) => `[${e.date_str} ${e.time_str}] [${e.severity.toUpperCase()}] ${e.summary} (${Math.round(e.confidence * 100)}%)`
      ),
    ];
    content = lines.join("\n");
  }

  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `OpenVisionCam_Report_${dateTag}.${ext}`;
  a.click();
  URL.revokeObjectURL(url);
}

function switchCamera(camId) {
  activeCamId = camId;
  document.getElementById("live-stream").src = `/api/stream/${camId}?t=${Date.now()}`;
  loadZones();
}

async function pollSystem() {
  try {
    const res = await fetch("/api/system");
    const sys = await res.json();

    const provEl = document.getElementById("hud-provider");
    if (sys.directml_active) {
      provEl.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span><span>DirectML iGPU: ACTIVE</span>`;
      provEl.className = "px-3 py-1.5 rounded-xl bg-slate-900/90 border border-emerald-500/30 text-emerald-300 flex items-center gap-2";
    } else {
      provEl.innerHTML = `<span class="w-2 h-2 rounded-full bg-amber-400"></span><span>Ryzen CPU Mode</span>`;
      provEl.className = "px-3 py-1.5 rounded-xl bg-slate-900/90 border border-amber-500/30 text-amber-300 flex items-center gap-2";
    }

    document.getElementById("hud-latency").innerText = `Latency: ${sys.last_latency_ms} ms`;
    document.getElementById("hud-cpu").innerText = `CPU: ${sys.cpu_percent}% | RAM: ${sys.ram_percent}%`;

    const camStats = sys.cameras?.[activeCamId];
    if (camStats) {
      document.getElementById("hud-fps").innerText = camStats.fps;
      document.getElementById("hud-targets").innerText = camStats.active_targets;
      document.getElementById("cam-status-badge").innerText = camStats.status;
    }
  } catch (err) {
    console.warn("Telemetry poll error:", err);
  }
}

/* ==================== INTERACTIVE ZONE DRAWING ==================== */
async function loadZones() {
  const res = await fetch(`/api/zones/${activeCamId}`);
  const data = await res.json();
  currentZones = data.zones || [];
}

function startZoneDraw(mode) {
  drawingMode = mode;
  drawingPoints = [];
  canvas.style.pointerEvents = "auto";
  canvas.style.cursor = "crosshair";
  resizeCanvas();

  document.getElementById("draw-hint").classList.remove("hidden");
  document.getElementById("btn-finish-zone").classList.remove("hidden");
  document.getElementById("btn-cancel-zone").classList.remove("hidden");
}

function cancelZoneDraw() {
  drawingMode = null;
  drawingPoints = [];
  canvas.style.pointerEvents = "none";
  canvas.style.cursor = "default";
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  document.getElementById("draw-hint").classList.add("hidden");
  document.getElementById("btn-finish-zone").classList.add("hidden");
  document.getElementById("btn-cancel-zone").classList.add("hidden");
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
}

canvas.addEventListener("click", (e) => {
  if (!drawingMode) return;
  const rect = canvas.getBoundingClientRect();
  const xNorm = (e.clientX - rect.left) / rect.width;
  const yNorm = (e.clientY - rect.top) / rect.height;

  drawingPoints.push([Number(xNorm.toFixed(4)), Number(yNorm.toFixed(4))]);
  renderDraftShape();

  if (drawingMode === "tripwire" && drawingPoints.length === 2) {
    finishZoneDraw();
  }
});

function renderDraftShape() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (drawingPoints.length === 0) return;

  const color = drawingMode === "tripwire" ? "#ff007f" : "#00f3ff";
  ctx.strokeStyle = color;
  ctx.fillStyle = "rgba(0, 243, 255, 0.2)";
  ctx.lineWidth = 2.5;

  ctx.beginPath();
  drawingPoints.forEach(([nx, ny], idx) => {
    const px = nx * canvas.width;
    const py = ny * canvas.height;
    if (idx === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });

  if (drawingMode === "polygon" && drawingPoints.length >= 3) {
    ctx.closePath();
    ctx.fill();
  }
  ctx.stroke();

  drawingPoints.forEach(([nx, ny]) => {
    ctx.beginPath();
    ctx.arc(nx * canvas.width, ny * canvas.height, 5, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
  });
}

async function finishZoneDraw() {
  if (!drawingMode) return;
  if (drawingMode === "polygon" && drawingPoints.length < 3) {
    alert("Click at least 3 points to create a polygon intrusion zone.");
    return;
  }
  if (drawingMode === "tripwire" && drawingPoints.length < 2) {
    alert("Click 2 points to create a perimeter tripwire.");
    return;
  }

  const label = prompt(
    `Name for this ${drawingMode}:`,
    drawingMode === "tripwire" ? `Tripwire ${currentZones.length + 1}` : `Zone ${currentZones.length + 1}`
  );
  if (!label) {
    cancelZoneDraw();
    return;
  }

  currentZones.push({
    id: `z_${Date.now()}`,
    name: label,
    type: drawingMode,
    color: drawingMode === "tripwire" ? "#ff007f" : "#00f3ff",
    points: drawingPoints,
    enabled: true,
  });

  await fetch(`/api/zones/${activeCamId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(currentZones),
  });

  cancelZoneDraw();
}

async function clearCameraZones() {
  currentZones = [];
  await fetch(`/api/zones/${activeCamId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify([]),
  });
}

/* ==================== NATURAL LANGUAGE SEARCH & EVENTS ==================== */
function debouncedSearch() {
  clearTimeout(searchTimeout);
  searchTimeout = setTimeout(loadEvents, 250);
}

function quickSearch(term) {
  document.getElementById("nlp-search-input").value = term;
  loadEvents();
}

async function loadEvents() {
  const q = document.getElementById("nlp-search-input").value;
  const severity = document.getElementById("severity-filter").value;
  const url = `/api/events?q=${encodeURIComponent(q)}&severity=${encodeURIComponent(severity)}&limit=60`;

  const res = await fetch(url);
  const data = await res.json();
  const grid = document.getElementById("events-grid");

  if (data.disabled) {
    grid.innerHTML = `<div class="col-span-full text-center py-8 text-xs font-mono text-slate-500">Natural Language Search is toggled OFF.</div>`;
    return;
  }

  const events = data.events || [];
  if (events.length === 0) {
    grid.innerHTML = `<div class="col-span-full text-center py-8 text-xs font-mono text-slate-500">No matching events logged yet.</div>`;
    return;
  }

  // Trigger subtle audio ping on new alert if sound is enabled
  const topAlert = events.find((e) => e.severity === "alert");
  if (topAlert && topAlert.id !== lastKnownAlertId) {
    if (lastKnownAlertId !== null && soundAlertsEnabled) {
      playAlertPing();
    }
    lastKnownAlertId = topAlert.id;
  }

  grid.innerHTML = events
    .map((evt) => {
      const isAlert = evt.severity === "alert";
      const thumbUrl = evt.thumbnail ? `/api/thumbnails/${evt.thumbnail}` : "";
      const hasClip = Boolean(evt.clip);

      return `
      <div class="rounded-xl bg-slate-900/75 border ${isAlert ? "border-pink-500/40" : "border-white/10"} overflow-hidden flex flex-col justify-between">
        <div>
          ${
            thumbUrl
              ? `<div class="relative h-32 bg-black overflow-hidden">
                   <img src="${thumbUrl}" loading="lazy" class="w-full h-full object-cover" />
                   <span class="absolute top-2 left-2 text-[10px] font-mono px-2 py-0.5 rounded ${isAlert ? "bg-pink-600 text-white" : "bg-cyan-500/80 text-slate-950 font-bold"}">
                     ${isAlert ? "⚠ ALERT" : evt.label.toUpperCase()}
                   </span>
                   <span class="absolute bottom-1.5 right-2 text-[10px] font-mono bg-black/75 px-1.5 py-0.5 rounded text-slate-200">
                     ${evt.time_str}
                   </span>
                 </div>`
              : ""
          }
          <div class="p-3 space-y-1">
            <div class="text-xs font-semibold text-white leading-snug">${evt.summary}</div>
            <div class="text-[11px] font-mono text-slate-400 flex items-center justify-between">
              <span>${Math.round(evt.confidence * 100)}% conf</span>
              <span>${evt.dwell_seconds ? `${evt.dwell_seconds}s dwell` : evt.camera_name}</span>
            </div>
          </div>
        </div>
        ${
          hasClip
            ? `<div class="px-3 pb-3 flex items-center gap-2">
                 <button onclick="openClipModal('${evt.clip}', '${evt.summary.replace(/'/g, "")}')" class="flex-1 py-1.5 rounded-lg bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-400/40 text-cyan-200 text-[11px] font-mono font-semibold transition">
                   ▶ Play DVR Clip
                 </button>
                 <a href="/api/clips/${evt.clip}" download class="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-[11px] font-mono">
                   ⬇
                 </a>
               </div>`
            : ""
        }
      </div>`;
    })
    .join("");
}

async function clearAllEvents() {
  await fetch("/api/events", { method: "DELETE" });
  loadEvents();
  loadSummary();
}

/* ==================== AI BRIEFING ==================== */
async function loadSummary() {
  const res = await fetch("/api/summary");
  const data = await res.json();
  if (data.disabled) {
    document.getElementById("summary-headline").innerText = "AI Briefings Disabled";
    document.getElementById("summary-narrative").innerText = "Enable 'AI Incident Briefings' toggle above to view automated summaries.";
    document.getElementById("summary-highlights").innerHTML = "";
    return;
  }

  document.getElementById("summary-headline").innerText = data.headline || "";
  document.getElementById("summary-narrative").innerText = data.narrative || "";
  document.getElementById("summary-highlights").innerHTML = (data.highlights || [])
    .map((h) => `<li class="border-l-2 border-cyan-500/50 pl-2">${h}</li>`)
    .join("");
}

/* ==================== MODALS & AUDIO ==================== */
function openCameraModal() {
  document.getElementById("camera-modal").classList.remove("hidden");
}
function closeCameraModal() {
  document.getElementById("camera-modal").classList.add("hidden");
}

async function saveCamera() {
  const id = document.getElementById("cam-id-input").value.trim() || `cam-${Date.now()}`;
  const name = document.getElementById("cam-name-input").value.trim() || id;
  const source = document.getElementById("cam-src-input").value.trim();
  if (!source) {
    alert("Please enter an RTSP URL, webcam:0, or demo:simulated");
    return;
  }

  await fetch("/api/cameras", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, name, source, enabled: true }),
  });

  closeCameraModal();
  await loadCameras();
  switchCamera(id);
}

function openClipModal(filename, title) {
  document.getElementById("clip-modal-title").innerText = title || "DVR Replay";
  const video = document.getElementById("clip-video");
  video.src = `/api/clips/${filename}`;
  document.getElementById("clip-modal").classList.remove("hidden");
}

function closeClipModal() {
  const video = document.getElementById("clip-video");
  video.pause();
  video.src = "";
  document.getElementById("clip-modal").classList.add("hidden");
}

function playAlertPing() {
  try {
    const actx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = actx.createOscillator();
    const gain = actx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(880, actx.currentTime);
    gain.gain.setValueAtTime(0.08, actx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, actx.currentTime + 0.22);
    osc.connect(gain);
    gain.connect(actx.destination);
    osc.start();
    osc.stop(actx.currentTime + 0.22);
  } catch (e) {}
}

window.addEventListener("DOMContentLoaded", init);

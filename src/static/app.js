/* FoulFilter batch UI */
const $ = (sel) => document.querySelector(sel);

const dropzone = $("#dropzone");
const fileInput = $("#file-input");
const uploadBtn = $("#upload-btn");
const zipBtn = $("#zip-btn");
const jobsBody = $("#jobs-body");

let selectedFiles = [];
const rows = new Map(); // job_id -> tr element
let historyMetrics = []; // historical job timings for ETA
const jobState = new Map(); // job_id -> {prevProgress, stageStarted, clipDuration}

// --- init ---------------------------------------------------------------
fetch("/config").then(r => r.json()).then(cfg => {
  $("#upload-limit").textContent = `up to ${cfg.max_upload_mb} MB per file`;
  if (cfg.ai_enhance) $("#ai-badge").hidden = false;
}).catch(() => {});

fetch("/metrics").then(r => r.json()).then(m => {
  historyMetrics = m || [];
}).catch(() => {});

fetch("/jobs").then(r => r.json()).then(jobs => {
  for (const job of jobs) renderRow(job);
}).catch(() => {});

connectEvents();

// Refresh ETA labels every 5s (ETAs move as time passes even with no SSE events).
setInterval(() => {
  let anyActive = false;
  for (const tr of rows.values()) {
    const id = tr.dataset.jobId;
    const st = jobState.get(id);
    if (st && st.startedAt && st.active) {
      anyActive = true;
      const eta = computeEta(id);
      const etaEl = tr.querySelector(".eta");
      if (etaEl) etaEl.textContent = formatEta(eta);
    }
  }
  if (anyActive) updateZipVisibility();
}, 5000);

// --- upload selection ----------------------------------------------------
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag");
  setFiles(e.dataTransfer.files);
});
fileInput.addEventListener("change", () => setFiles(fileInput.files));

function setFiles(list) {
  selectedFiles = Array.from(list);
  uploadBtn.disabled = selectedFiles.length === 0;
  const inner = $("#drop-inner");
  inner.firstElementChild.textContent = selectedFiles.length
    ? `${selectedFiles.length} file(s) selected`
    : "Drop files here";
}

uploadBtn.addEventListener("click", async () => {
  if (!selectedFiles.length) return;
  const form = new FormData();
  for (const f of selectedFiles) form.append("files", f, f.name);
  form.append("censor_method", $("#censor-method").value);
  form.append("debug", $("#opt-debug").checked);
  form.append("rescan", $("#opt-rescan").checked);

  uploadBtn.disabled = true;
  try {
    const res = await fetch("/upload", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed");
    toast(`Queued ${data.job_ids.length} job(s)`, "ok");
    selectedFiles = [];
    fileInput.value = "";
    setFiles([]);
  } catch (err) {
    toast(err.message, "error");
  } finally {
    uploadBtn.disabled = false;
  }
});

zipBtn.addEventListener("click", () => {
  const ids = completedCheckedIds();
  if (!ids.length) { toast("Select completed downloads first", "error"); return; }
  window.location.href = `/download_zip?ids=${ids.join(",")}`;
});

function completedCheckedIds() {
  return [...rows.entries()]
    .filter(([_, tr]) => tr.querySelector(".pick")?.checked)
    .map(([id]) => id);
}

// --- SSE ------------------------------------------------------------------
function connectEvents() {
  const es = new EventSource("/events");
  es.onmessage = (e) => {
    try { renderRow(JSON.parse(e.data)); } catch { /* ignore malformed */ }
  };
  es.onerror = () => { es.close(); setTimeout(connectEvents, 2000); };
}

// --- ETA helpers -----------------------------------------------------------
const STAGE_WEIGHTS = { transcribing: 0.225, matching: 0.50, aligning: 0.65, refining: 0.825, editing: 0.90, completed: 1.0 };

function trackStage(job) {
  const st = jobState.get(job.id);
  if (!st) return;
  const pct = job.progress ?? 0;
  if (job.started_at && pct > 0 && !st.startedAt) st.startedAt = job.started_at * 1000;
  let stage = job.stage;
  if (pct >= 45 && stage === "queued") stage = "transcribing";
  st.currentStage = stage;
  st.active = job.status === "processing";
  st.progress = pct;
}

function computeEta(jobId) {
  const st = jobState.get(jobId);
  if (!st || !st.startedAt) return null;
  const pct = st.progress ?? 0;
  if (pct <= 0 || pct >= 100) return null;
  const elapsed = (Date.now() - st.startedAt) / 1000;
  const clip = st.clipDuration || 120;
  const stageName = st.currentStage || "transcribing";
  return estimateRemaining(clip, stageName, pct, elapsed);
}

function estimateRemaining(clipSeconds, currentStage, progressPct, elapsedSecs) {
  const progressFrac = progressPct / 100;
  if (progressFrac <= 0 || progressFrac >= 1) return null;
  const lo = clipSeconds * 0.5, hi = clipSeconds * 1.5;
  const similar = historyMetrics.filter(m => m.clip_seconds >= lo && m.clip_seconds <= hi);
  if (similar.length > 0) return estimateFromHistory(similar, clipSeconds, currentStage, progressFrac, elapsedSecs);
  const totalEst = elapsedSecs / Math.max(0.01, progressFrac);
  return Math.max(0, totalEst * (1 - progressFrac));
}

function estimateFromHistory(similar, clipSeconds, currentStage, progressFrac, elapsedSecs) {
  const stageOrder = ["transcribing", "matching", "aligning", "refining", "editing"];
  const curIdx = stageOrder.indexOf(currentStage);
  const ratios = [];
  for (const m of similar) {
    const stages = m.stages;
    let remaining = 0;
    if (curIdx >= 0) {
      const curStageTime = stages[currentStage] || 0;
      if (curStageTime > 0) {
        const prevMid = curIdx > 0 ? STAGE_WEIGHTS[stageOrder[curIdx - 1]] : 0;
        const range = Math.max((STAGE_WEIGHTS[currentStage] || progressFrac) - prevMid, 0.01);
        const done = Math.min(1, Math.max(0, (progressFrac - prevMid) / range));
        remaining += curStageTime * (1 - done);
      }
      for (let i = curIdx + 1; i < stageOrder.length; i++) remaining += stages[stageOrder[i]] || 0;
    }
    if (remaining > 0) {
      const total = Object.values(stages).reduce((a, b) => a + b, 0);
      ratios.push(remaining / Math.max(0.01, total));
    }
  }
  if (ratios.length === 0) return null;
  const avg = ratios.reduce((a, b) => a + b, 0) / ratios.length;
  return Math.max(0, clipSeconds * avg);
}

function formatEta(secs) {
  if (secs == null || secs < 0) return "";
  secs = Math.round(secs);
  if (secs < 5) return "";
  if (secs < 60) return `~${secs}s left`;
  return `~${Math.floor(secs / 60)}m ${secs % 60}s left`;
}

// --- table rendering -------------------------------------------------------
function renderRow(job) {
  let tr = rows.get(job.id);
  if (!tr) {
    tr = document.createElement("tr");
    tr.dataset.jobId = job.id;
    tr.innerHTML = `
      <td class="name"></td>
      <td><span class="chip"></span></td>
      <td>
        <div class="progress-track"><div class="progress-bar"></div></div>
        <span class="stage"></span>
        <span class="eta"></span>
      </td>
      <td class="row-actions" style="white-space:nowrap; text-align:right;"></td>`;
    jobsBody.prepend(tr);
    rows.set(job.id, tr);
  }

  if (!jobState.has(job.id)) {
    jobState.set(job.id, {
      startedAt: null,
      currentStage: null,
      active: false,
      progress: 0,
      clipDuration: 120,
    });
  }

  // Seed startedAt from the backend-provided timestamp if present.
  const st = jobState.get(job.id);
  if (job.started_at && !st.startedAt) st.startedAt = job.started_at * 1000;

  tr.querySelector(".name").textContent = job.filename || job.id;
  const chip = tr.querySelector(".chip");
  chip.textContent = job.status;
  chip.className = `chip ${job.status}`;

  const bar = tr.querySelector(".progress-bar");
  bar.style.width = `${job.progress ?? 0}%`;
  bar.style.background =
    job.status === "failed" ? "var(--err)" :
    job.status === "completed" ? "var(--ok)" : "var(--accent)";

  const stageEl = tr.querySelector(".stage");
  const etaEl = tr.querySelector(".eta");

  if (job.status === "queued") {
    stageEl.textContent = job.detail
      ? `${job.stage || ""} — ${job.detail}`.replace(/^— /, "") : (job.stage || "");
    etaEl.textContent = "";
  } else if (job.status === "processing") {
    trackStage(job);
    const eta = computeEta(job.id);
    stageEl.textContent = job.detail
      ? `${job.stage || ""} — ${job.detail}`.replace(/^— /, "") : (job.stage || "");
    etaEl.textContent = formatEta(eta);
  } else {
    trackStage(job);
    stageEl.textContent = job.detail
      ? `${job.stage || ""} — ${job.detail}`.replace(/^— /, "") : (job.stage || "");
    etaEl.textContent = "";
  }

  const actions = tr.querySelector(".row-actions");
  actions.innerHTML = "";
  if (job.status === "queued" || job.status === "processing") {
    actions.appendChild(btn("Cancel", "cancel", () =>
      fetch(`/jobs/${job.id}`, { method: "DELETE" })));
  } else {
    const pick = Object.assign(document.createElement("input"), {
      type: "checkbox", className: "pick",
    });
    pick.checked = job.status === "completed";
    pick.disabled = job.status !== "completed";
    pick.addEventListener("change", updateZipVisibility);
    actions.appendChild(pick);
    if (job.status === "completed") {
      actions.appendChild(btn("Download", "dl", () =>
        window.location.href = `/download/${job.id}`));
    }
    actions.appendChild(btn("Remove", "cancel", async () => {
      await fetch(`/jobs/${job.id}`, { method: "DELETE" });
      tr.remove(); rows.delete(job.id); updateZipVisibility();
    }));
  }
  updateZipVisibility();
}

function btn(label, cls, onClick) {
  const b = document.createElement("button");
  b.textContent = label;
  b.classList.add(cls === "dl" ? "" : "secondary", cls);
  b.addEventListener("click", onClick);
  return b;
}

function updateZipVisibility() {
  zipBtn.hidden = completedCheckedIds().length === 0;
}

// --- toasts -----------------------------------------------------------------
function toast(msg, kind = "") {
  const t = document.createElement("div");
  t.className = `toast ${kind}`;
  t.textContent = msg;
  $("#toasts").appendChild(t);
  setTimeout(() => t.remove(), 6000);
}

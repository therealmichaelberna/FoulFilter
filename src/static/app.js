/* FoulFilter batch UI */
const $ = (sel) => document.querySelector(sel);

const dropzone = $("#dropzone");
const fileInput = $("#file-input");
const uploadBtn = $("#upload-btn");
const zipBtn = $("#zip-btn");
const jobsBody = $("#jobs-body");

let selectedFiles = [];
const rows = new Map(); // job_id -> tr element

// --- init ---------------------------------------------------------------
fetch("/config").then(r => r.json()).then(cfg => {
  $("#upload-limit").textContent = `up to ${cfg.max_upload_mb} MB per file`;
  if (cfg.ai_enhance) $("#ai-badge").hidden = false;
}).catch(() => {});

fetch("/jobs").then(r => r.json()).then(jobs => {
  for (const job of jobs) renderRow(job);
}).catch(() => {});

connectEvents();

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
      </td>
      <td class="row-actions" style="white-space:nowrap; text-align:right;"></td>`;
    jobsBody.prepend(tr);
    rows.set(job.id, tr);
  }

  tr.querySelector(".name").textContent = job.filename || job.id;
  const chip = tr.querySelector(".chip");
  chip.textContent = job.status;
  chip.className = `chip ${job.status}`;

  const bar = tr.querySelector(".progress-bar");
  bar.style.width = `${job.progress ?? 0}%`;
  bar.style.background =
    job.status === "failed" ? "var(--err)" :
    job.status === "completed" ? "var(--ok)" : "var(--accent)";
  tr.querySelector(".stage").textContent = job.detail
    ? `${job.stage || ""} — ${job.detail}`.replace(/^— /, "") : (job.stage || "");

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

const tabs = {
  perturb: { title: "Perturb", configs: ["pipeline"] },
  diffuse: { title: "Diffuse", configs: ["pipeline"] },
  pipeline: { title: "Pipeline", configs: ["pipeline"] },
  brute: { title: "Brute Force", configs: ["pipeline", "brute"] },
  batch_brute: { title: "Batch Brute Force", configs: ["pipeline", "brute", "batch_brute"] },
  history: { title: "History", configs: [] },
};

let activeTab = "perturb";
let baseConfigs = {};
let activeRunId = null;
let selectedRunId = null;
let eventSource = null;
const modelStates = new Map();

const el = (id) => document.getElementById(id);
const fileUrl = (path) => `/api/file?path=${encodeURIComponent(path)}`;

function setStatus(text) {
  el("connection-status").textContent = text;
}

async function apiJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail || response.statusText);
  }
  return response.json();
}

async function loadConfigs() {
  baseConfigs = await apiJson("/api/configs");
  renderConfigPanel();
  renderDeepFaceDefaults();
}

function renderConfigPanel() {
  const panel = el("config-panel");
  panel.innerHTML = "";
  const configNames = tabs[activeTab].configs;
  if (!configNames.length) {
    panel.innerHTML = "<p>Select a run in history to inspect its saved config and report.</p>";
    return;
  }
  for (const name of configNames) {
    const block = document.createElement("div");
    block.className = "config-block";
    block.innerHTML = `<h3>${name}.json temporary override</h3>`;
    const textarea = document.createElement("textarea");
    textarea.id = `config-${name}`;
    textarea.spellcheck = false;
    textarea.value = JSON.stringify(baseConfigs[name], null, 2);
    block.appendChild(textarea);
    panel.appendChild(block);
  }
}

function readConfigEditors() {
  const configs = {};
  for (const name of tabs[activeTab].configs) {
    const textarea = el(`config-${name}`);
    configs[name] = JSON.parse(textarea.value);
  }
  return configs;
}

function resetRunView() {
  el("run-id").textContent = "none";
  el("run-status").textContent = "idle";
  el("current-attempt").textContent = "-";
  el("current-mean").textContent = "-";
  el("count-success").textContent = "0";
  el("count-unsuccessful").textContent = "0";
  el("count-failures").textContent = "0";
  el("count-skip-resume").textContent = "0 / 0";
  el("min-score").textContent = "-";
  el("max-score").textContent = "-";
  el("combo-state").textContent = "-";
  el("events").innerHTML = "";
  el("inspector").textContent = "{}";
  modelStates.clear();
  renderModels();
  for (const key of ["original", "perturbed", "original_diffused", "perturbed_diffused"]) {
    setImageBox(key, null);
  }
}

function setImageBox(name, path, loading = false) {
  const mapped = name === "diffused" ? "original_diffused" : name;
  const box = el(`img-${mapped}`);
  if (!box) return;
  box.classList.toggle("loading", loading);
  if (loading) {
    box.textContent = "";
    return;
  }
  if (!path) {
    box.textContent = "empty";
    box.innerHTML = "empty";
    return;
  }
  box.innerHTML = `<img src="${fileUrl(path)}&t=${Date.now()}" alt="${mapped}" />`;
}

function renderDeepFaceDefaults() {
  const models = baseConfigs.pipeline?.deepface?.models || {};
  modelStates.clear();
  for (const [name, enabled] of Object.entries(models)) {
    if (enabled) modelStates.set(name, { status: "pending", value: "-" });
  }
  renderModels();
}

function renderModels() {
  const container = el("deepface-models");
  container.innerHTML = "";
  for (const [name, state] of modelStates.entries()) {
    const div = document.createElement("div");
    div.className = `model ${state.status || "pending"}`;
    div.innerHTML = `<span>${name}</span><strong>${state.value || state.status}</strong>`;
    container.appendChild(div);
  }
  if (!modelStates.size) {
    container.innerHTML = "<p>No enabled DeepFace models for the current view.</p>";
  }
}

function addEventLine(event) {
  const line = document.createElement("div");
  line.className = "event-line";
  const label = event.type || "event";
  const suffix = event.run_number !== undefined ? ` run ${event.run_number}` : "";
  line.textContent = `${event.sequence || ""} ${label}${suffix}`;
  el("events").prepend(line);
}

function updateSummary(summary = {}) {
  el("count-success").textContent = summary.successful ?? el("count-success").textContent;
  el("count-unsuccessful").textContent = summary.unsuccessful ?? el("count-unsuccessful").textContent;
  el("count-failures").textContent = summary.failures ?? el("count-failures").textContent;
  const skipped = summary.skipped_runs ?? summary.skipped ?? 0;
  const resumed = summary.resumed_runs ?? summary.resumed ?? 0;
  el("count-skip-resume").textContent = `${skipped} / ${resumed}`;
}

function updateScorePanels(event) {
  if (event.min_score) el("min-score").textContent = JSON.stringify(event.min_score, null, 2);
  if (event.max_score) el("max-score").textContent = JSON.stringify(event.max_score, null, 2);
}

function handleEvent(event) {
  addEventLine(event);
  if (event.type?.includes("failed") || event.type === "deepface_model_error") setStatus("error");
  if (event.status) el("run-status").textContent = event.status;
  if (event.run_number !== undefined) el("current-attempt").textContent = event.run_number;
  if (event.summary) updateSummary(event.summary);

  if (event.type === "diffusion_started") {
    setImageBox("original_diffused", null, true);
    if (activeTab !== "diffuse") setImageBox("perturbed_diffused", null, true);
  }
  if (event.type === "image_written" && event.name && event.path) {
    setImageBox(event.name, event.path);
  }
  if (event.type === "deepface_model_pending") {
    modelStates.set(event.model, { status: "pending", value: "pending" });
    renderModels();
  }
  if (event.type === "deepface_model_running") {
    modelStates.set(event.model, { status: "running", value: "running" });
    renderModels();
  }
  if (event.type === "deepface_model_completed") {
    const value = event.percentage === null || event.percentage === undefined ? "-" : `${event.percentage.toFixed(2)}%`;
    modelStates.set(event.model, { status: "completed", value });
    renderModels();
  }
  if (event.type === "deepface_model_error") {
    modelStates.set(event.model, { status: "error", value: "error" });
    renderModels();
  }
  if (event.type === "running_mean_updated") {
    el("current-mean").textContent = `${event.mean_match_percent.toFixed(2)}%`;
  }
  if (event.type === "min_max_score_updated") {
    updateScorePanels(event);
  }
  if (event.type?.startsWith("batch_combo_") || event.image_path || event.prompt) {
    el("combo-state").textContent = JSON.stringify({
      type: event.type,
      image_index: event.image_index,
      prompt_index: event.prompt_index,
      image_path: event.image_path,
      prompt: event.prompt,
      status: event.status,
    }, null, 2);
  }
  if (event.type === "ui_run_completed" || event.type === "ui_run_failed") {
    el("stop-btn").disabled = true;
    refreshHistory();
  }
}

function connectEvents(runId) {
  if (eventSource) eventSource.close();
  eventSource = new EventSource(`/api/runs/${runId}/events`);
  eventSource.onopen = () => setStatus("streaming");
  eventSource.onerror = () => setStatus("stream closed");
  eventSource.onmessage = (message) => {
    const event = JSON.parse(message.data);
    handleEvent(event);
  };
}

async function startActiveRun() {
  if (activeTab === "history") return;
  try {
    const configs = readConfigEditors();
    resetRunView();
    renderDeepFaceDefaults();
    const result = await apiJson(`/api/runs/${activeTab}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ configs }),
    });
    activeRunId = result.run_id;
    selectedRunId = result.run_id;
    el("run-id").textContent = result.run_id;
    el("run-status").textContent = result.status;
    el("stop-btn").disabled = false;
    el("resume-btn").disabled = true;
    connectEvents(result.run_id);
    refreshHistory();
  } catch (error) {
    setStatus("start failed");
    alert(error.message);
  }
}

async function stopActiveRun() {
  if (!activeRunId) return;
  await apiJson(`/api/runs/${activeRunId}/stop`, { method: "POST" });
  el("run-status").textContent = "stopping";
}

async function resumeSelectedRun() {
  if (!selectedRunId) return;
  const result = await apiJson(`/api/runs/${selectedRunId}/resume`, { method: "POST" });
  activeRunId = result.run_id;
  el("run-id").textContent = result.run_id;
  el("run-status").textContent = result.status;
  el("stop-btn").disabled = false;
  connectEvents(result.run_id);
}

async function inspectRun(runId) {
  selectedRunId = runId;
  const [run, report, events] = await Promise.all([
    apiJson(`/api/runs/${runId}`),
    apiJson(`/api/runs/${runId}/report`),
    apiJson(`/api/runs/${runId}/events.json`),
  ]);
  el("run-id").textContent = run.run_id;
  el("run-status").textContent = run.status;
  el("inspector").textContent = JSON.stringify({ run, report }, null, 2);
  el("events").innerHTML = "";
  for (const event of events.slice(-250).reverse()) addEventLine(event);
  const outputs = report.outputs || {};
  setImageBox("original", outputs.original);
  setImageBox("perturbed", outputs.perturbed);
  setImageBox("original_diffused", outputs.original_diffused || outputs.diffused);
  setImageBox("perturbed_diffused", outputs.perturbed_diffused);
  if (run.progress_summary?.min_score || run.progress_summary?.max_score) {
    updateScorePanels(run.progress_summary);
  }
  const canResume = ["brute", "batch_brute"].includes(run.run_type) && !["queued", "running"].includes(run.status);
  el("resume-btn").disabled = !canResume;
}

async function refreshHistory() {
  const runs = await apiJson("/api/runs");
  const list = el("history-list");
  list.innerHTML = "";
  const filtered = activeTab === "history" ? runs : runs.filter((run) => run.run_type === activeTab);
  for (const run of filtered) {
    const item = document.createElement("div");
    item.className = "history-item";
    item.innerHTML = `
      <strong>${run.run_id}</strong>
      <span>${run.run_type}</span>
      <span class="pill ${run.status}">${run.status}</span>
      <span>${run.started_at}</span>
      <button data-run="${run.run_id}">Inspect</button>
    `;
    item.querySelector("button").addEventListener("click", () => inspectRun(run.run_id));
    list.appendChild(item);
  }
  if (!filtered.length) list.innerHTML = "<p>No runs yet for this view.</p>";
}

function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll(".nav").forEach((button) => button.classList.toggle("active", button.dataset.tab === tab));
  el("tab-title").textContent = tabs[tab].title;
  el("config-panel").style.display = tab === "history" ? "none" : "block";
  el("start-btn").disabled = tab === "history";
  renderConfigPanel();
  renderDeepFaceDefaults();
  refreshHistory();
}

document.querySelectorAll(".nav").forEach((button) => {
  button.addEventListener("click", () => switchTab(button.dataset.tab));
});
el("start-btn").addEventListener("click", startActiveRun);
el("stop-btn").addEventListener("click", stopActiveRun);
el("resume-btn").addEventListener("click", resumeSelectedRun);
el("refresh-history").addEventListener("click", refreshHistory);

loadConfigs()
  .then(refreshHistory)
  .catch((error) => {
    setStatus("config load failed");
    el("config-panel").textContent = error.message;
  });

"use strict";

const state = {
  configs: { tx: [], channel: [], simulation: [] },
  selected: { tx: "", channel: "", simulation: "" },
  texts: { tx: "", channel: "", simulation: "" },
  savedTexts: { tx: "", channel: "", simulation: "" },
  activeKind: "tx",
  currentJob: null,
  runs: [],
  pollTimer: null,
  rxResult: null,
  rxBusy: false,
  rxDefaults: {
    available: false,
    config_name: "",
    input_name: "",
    input_format: "matlab-h5",
    input_domain: "frequency",
    noise_variance: 0,
    channel_estimator: "dmrs",
    detector: "lmmse",
    detector_parameter: null,
    detector_damping: 0.25,
    max_delay_spread_s: 3e-6,
  },
};

const $ = (id) => document.getElementById(id);
const editor = $("config-editor");
const statusLabels = { queued: "排队中", running: "运行中", completed: "已完成", failed: "失败", cancelled: "已取消" };
const detectorColors = ["#315fdf", "#e97832", "#8b55d9", "#3b9b6c", "#d4a719", "#d34e70"];

function switchWorkflow(name) {
  const rxActive = name === "rx";
  $("bler-workflow").classList.toggle("hidden", rxActive);
  $("rx-workflow").classList.toggle("hidden", !rxActive);
  for (const [id, active] of [["bler-workflow-tab", !rxActive], ["rx-workflow-tab", rxActive]]) {
    $(id).classList.toggle("active", active);
    $(id).setAttribute("aria-selected", String(active));
  }
  if (rxActive && state.rxResult) requestAnimationFrame(() => drawConstellation(state.rxResult.constellation));
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  let payload;
  try { payload = await response.json(); } catch { payload = null; }
  if (!response.ok) {
    const detail = payload && (payload.detail || payload.error || payload.message);
    throw new Error(detail || `请求失败（HTTP ${response.status}）`);
  }
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error(`服务接口 ${path} 没有返回有效的 JSON 对象；请刷新页面并确认网页服务已更新。`);
  }
  return payload;
}

function showAlert(message, error = true) {
  $("alert-text").textContent = String(message);
  $("global-alert").classList.toggle("error", error);
  $("global-alert").classList.remove("hidden");
}

function formatTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(date);
}

function formatDuration(job) {
  if (!job || !job.started_at) return "—";
  const start = new Date(job.started_at).getTime();
  const end = job.finished_at ? new Date(job.finished_at).getTime() : Date.now();
  if (!Number.isFinite(start) || !Number.isFinite(end)) return "—";
  const seconds = Math.max(0, (end - start) / 1000);
  return seconds < 60 ? `${seconds.toFixed(1)} 秒` : `${Math.floor(seconds / 60)} 分 ${Math.round(seconds % 60)} 秒`;
}

function updateLines() {
  const count = Math.max(1, editor.value.split("\n").length);
  $("line-numbers").textContent = Array.from({ length: count }, (_, i) => i + 1).join("\n");
  $("line-numbers").scrollTop = editor.scrollTop;
}

function updateDirty() {
  const dirty = editor.value !== state.savedTexts[state.activeKind];
  $("dirty-indicator").classList.toggle("hidden", !dirty);
}

async function loadConfig(kind, name, force = false) {
  if (!name) {
    state.texts[kind] = "";
    state.savedTexts[kind] = "";
    if (state.activeKind === kind) editor.value = "";
    updateLines();
    return;
  }
  if (!force && state.selected[kind] === name && state.savedTexts[kind]) return;
  const result = await api(`/api/configs/${encodeURIComponent(kind)}/${encodeURIComponent(name)}`);
  state.selected[kind] = name;
  state.texts[kind] = result.text || "";
  state.savedTexts[kind] = result.text || "";
  if (state.activeKind === kind) {
    editor.value = state.texts[kind];
    updateLines();
    updateDirty();
  }
}

function populateProfileSelect() {
  const select = $("profile-select");
  select.replaceChildren();
  const names = state.configs[state.activeKind] || [];
  if (!names.length) {
    const option = document.createElement("option");
    option.textContent = "无可用配置";
    option.value = "";
    select.append(option);
    select.disabled = true;
  } else {
    names.forEach((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      select.append(option);
    });
    select.disabled = false;
    select.value = state.selected[state.activeKind] || names[0];
  }
}

async function initConfigs() {
  const configs = await api("/api/configs");
  ["tx", "channel", "simulation"].forEach((kind) => {
    state.configs[kind] = Array.isArray(configs[kind]) ? configs[kind] : [];
    state.selected[kind] = kind === "simulation" && state.configs[kind].includes("bler_smoke.toml")
      ? "bler_smoke.toml" : state.configs[kind][0] || "";
  });
  populateProfileSelect();
  await Promise.all(["tx", "channel", "simulation"].map((kind) => loadConfig(kind, state.selected[kind], true)));
  editor.value = state.texts[state.activeKind];
  updateLines();
  updateDirty();
}

async function switchKind(kind) {
  state.texts[state.activeKind] = editor.value;
  state.activeKind = kind;
  document.querySelectorAll(".config-tab").forEach((tab) => {
    const active = tab.dataset.kind === kind;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  populateProfileSelect();
  if (!state.savedTexts[kind] && state.selected[kind]) await loadConfig(kind, state.selected[kind], true);
  editor.value = state.texts[kind] || "";
  updateLines();
  updateDirty();
}

async function saveCurrentConfig() {
  const kind = state.activeKind;
  const name = state.selected[kind];
  if (!name) return showAlert("当前类型没有可保存的配置。", true);
  const button = $("save-button");
  button.disabled = true;
  try {
    const result = await api(`/api/configs/${encodeURIComponent(kind)}/${encodeURIComponent(name)}`, { method: "PUT", body: JSON.stringify({ text: editor.value }) });
    state.texts[kind] = result.text ?? editor.value;
    state.savedTexts[kind] = result.text ?? editor.value;
    editor.value = state.texts[kind];
    updateDirty();
    showAlert(`已保存 ${name}`, false);
  } catch (error) { showAlert(error.message); }
  finally { button.disabled = false; }
}

async function startRun() {
  state.texts[state.activeKind] = editor.value;
  if (!state.selected.tx || !state.selected.channel || !state.selected.simulation) return showAlert("请先为三类配置各选择一个文件。", true);
  const button = $("run-button");
  button.disabled = true;
  try {
    const job = await api("/api/runs", {
      method: "POST",
      body: JSON.stringify({
        tx_name: state.selected.tx,
        channel_name: state.selected.channel,
        simulation_name: state.selected.simulation,
        tx_text: state.texts.tx,
        channel_text: state.texts.channel,
        simulation_text: state.texts.simulation,
      }),
    });
    state.currentJob = job;
    renderJob(job);
    await loadRuns();
    schedulePoll(250);
  } catch (error) { showAlert(error.message); }
  finally { button.disabled = false; }
}

function renderJob(job) {
  if (!job) return;
  state.currentJob = job;
  const active = job.status === "queued" || job.status === "running";
  $("status-pill").textContent = statusLabels[job.status] || job.status || "未知状态";
  $("status-pill").className = `status-pill ${job.status || "idle"}`;
  $("cancel-button").classList.toggle("hidden", !active);
  $("job-id").textContent = job.id || "—";
  $("job-created").textContent = formatTime(job.created_at);
  $("job-duration").textContent = formatDuration(job);
  const allPoints = Array.isArray(job.points) ? job.points : [];
  const skippedCount = allPoints.filter((point) => point && point.skipped).length;
  const pointCount = allPoints.length - skippedCount;
  const totalPoints = Number(job.total_points) || 0;
  const donePoints = pointCount + skippedCount;
  $("job-points").textContent = totalPoints
    ? `${pointCount} / ${totalPoints}${skippedCount ? `（跳过 ${skippedCount}）` : ""}`
    : String(pointCount);
  $("progress-bar").style.width = totalPoints
    ? `${Math.min(100, donePoints / totalPoints * 100)}%`
    : job.status === "completed" ? "100%" : "0";
  for (const [key, url] of [["download-csv", job.output_csv], ["download-json", job.output_manifest]]) {
    const link = $(key);
    link.classList.toggle("hidden", !url);
    if (url) link.href = url;
  }
  const logs = Array.isArray(job.logs) ? job.logs : [];
  $("log-view").textContent = logs.length ? logs.join("\n") : active ? "任务已提交，等待日志…" : (job.error || "暂无运行日志。");
  $("log-view").scrollTop = $("log-view").scrollHeight;
  if (job.error) showAlert(job.error);
  renderResults(allPoints);
  updateActiveCount();
}

async function pollCurrentJob() {
  if (!state.currentJob || !state.currentJob.id) return;
  try {
    const job = await api(`/api/runs/${encodeURIComponent(state.currentJob.id)}`);
    renderJob(job);
    if (job.status === "queued" || job.status === "running") schedulePoll(2000);
    else await loadRuns();
  } catch (error) {
    showAlert(error.message);
    schedulePoll(4000);
  }
}

function schedulePoll(delay) {
  clearTimeout(state.pollTimer);
  state.pollTimer = setTimeout(pollCurrentJob, delay);
}

async function cancelRun() {
  if (!state.currentJob) return;
  $("cancel-button").disabled = true;
  try {
    const job = await api(`/api/runs/${encodeURIComponent(state.currentJob.id)}/cancel`, { method: "POST", body: "{}" });
    renderJob(job);
    await loadRuns();
  } catch (error) { showAlert(error.message); }
  finally { $("cancel-button").disabled = false; }
}

function updateActiveCount() {
  const active = state.runs.filter((run) => run.status === "queued" || run.status === "running").length;
  $("active-count").textContent = `${active} 个任务`;
}

async function loadRuns() {
  try {
    const payload = await api("/api/runs");
    state.runs = Array.isArray(payload.runs) ? payload.runs : [];
    renderHistory();
    updateActiveCount();
    if (!state.currentJob) {
      const first = state.runs.find((run) => run.status === "running" || run.status === "queued") || state.runs[0];
      if (first) {
        const job = await api(`/api/runs/${encodeURIComponent(first.id)}`);
        renderJob(job);
        if (job.status === "queued" || job.status === "running") schedulePoll(2000);
      }
    }
  } catch (error) { showAlert(error.message); }
}

function renderHistory() {
  const container = $("history-list");
  container.replaceChildren();
  if (!state.runs.length) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "尚无运行记录。";
    container.append(empty);
    return;
  }
  state.runs.forEach((run) => {
    const row = document.createElement("div");
    row.className = "history-item";
    row.tabIndex = 0;
    row.setAttribute("role", "button");
    const cells = [
      ["任务", run.id || "—", "history-id"],
      ["状态", statusLabels[run.status] || run.status || "—"],
      ["创建时间", formatTime(run.created_at)],
      ["配置", summarizeConfigs(run.config_names)],
    ];
    cells.forEach(([label, value, className]) => {
      const cell = document.createElement("div");
      const labelEl = document.createElement("span");
      labelEl.className = "history-label";
      labelEl.textContent = label;
      const valueEl = document.createElement("span");
      valueEl.className = className || "history-value";
      valueEl.textContent = value;
      cell.append(labelEl, valueEl);
      row.append(cell);
    });
    const arrow = document.createElement("span");
    arrow.className = "history-arrow";
    arrow.textContent = "›";
    row.append(arrow);
    const open = () => openRun(run.id);
    row.addEventListener("click", open);
    row.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); open(); } });
    container.append(row);
  });
}

function summarizeConfigs(names) {
  if (!names) return "—";
  return [names.tx, names.channel, names.simulation].filter(Boolean).join(" · ") || "—";
}

async function openRun(id) {
  try {
    const job = await api(`/api/runs/${encodeURIComponent(id)}`);
    renderJob(job);
    document.querySelector(".run-panel").scrollIntoView({ behavior: "smooth", block: "start" });
    if (job.status === "queued" || job.status === "running") schedulePoll(2000);
    else clearTimeout(state.pollTimer);
  } catch (error) { showAlert(error.message); }
}

function formatMetric(value, digits = 4) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  if (number !== 0 && Math.abs(number) < .001) return number.toExponential(2);
  return number.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
}

async function initRxConfigs() {
  const [defaultsResult, configsResult] = await Promise.allSettled([
    api("/api/rx/defaults"),
    api("/api/rx/configs"),
  ]);
  if (configsResult.status === "rejected") throw configsResult.reason;
  if (defaultsResult.status === "fulfilled" && defaultsResult.value && typeof defaultsResult.value === "object") {
    state.rxDefaults = { ...state.rxDefaults, ...defaultsResult.value };
  }
  const payload = configsResult.value;
  const configs = Array.isArray(payload.configs) ? payload.configs : [];
  const select = $("rx-config-select");
  select.replaceChildren();
  if (!configs.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "无可用 RX 配置";
    select.append(option);
    select.disabled = true;
  } else {
    configs.forEach((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      select.append(option);
    });
    if (state.rxDefaults.config_name && !configs.includes(state.rxDefaults.config_name)) {
      const option = document.createElement("option");
      option.value = state.rxDefaults.config_name;
      option.textContent = state.rxDefaults.config_name;
      select.append(option);
    }
    select.disabled = false;
    select.value = state.rxDefaults.config_name && [...select.options].some((option) => option.value === state.rxDefaults.config_name)
      ? state.rxDefaults.config_name : select.options[0]?.value || "";
    await loadRxConfig(select.value);
  }
  const defaultToggle = $("rx-use-default");
  defaultToggle.disabled = !state.rxDefaults.available;
  defaultToggle.checked = Boolean(state.rxDefaults.available);
  $("rx-input-domain").value = state.rxDefaults.input_domain || "frequency";
  $("rx-detector").value = state.rxDefaults.detector || "lmmse";
  $("rx-detector-parameter").value = state.rxDefaults.detector_parameter ?? "";
  $("rx-detector-damping").value = String(state.rxDefaults.detector_damping ?? 0.25);
  $("rx-detector").dispatchEvent(new Event("change"));
  $("rx-default-meta").textContent = state.rxDefaults.available
    ? `${state.rxDefaults.input_name || "本地默认夹具"} · ${state.rxDefaults.config_name || "默认配置"}`
    : "未发现本地默认夹具，请选择上传文件";
  setRxInputMode(defaultToggle.checked);
}

async function loadRxConfig(name) {
  if (!name) return;
  const config = await api(`/api/configs/rx/${encodeURIComponent(name)}`);
  $("rx-config-editor").value = config.text || "";
}

function formatFileSize(bytes) {
  const size = Number(bytes);
  if (!Number.isFinite(size) || size < 0) return "未知大小";
  if (size < 1024) return `${size} B`;
  if (size < 1024 ** 2) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 ** 2).toFixed(1)} MB`;
}

function selectedInputFormat(file) {
  const extension = String(file?.name || "").split(".").pop().toLowerCase();
  if (extension === "npz") return "npz";
  if (extension === "h5" || extension === "hdf5") return "matlab-h5";
  return "";
}

function isUsingDefaultRxCapture() {
  return Boolean(state.rxDefaults.available && $("rx-use-default").checked);
}

function setRxInputMode(useDefault) {
  const defaultMode = Boolean(useDefault && state.rxDefaults.available);
  const fileInput = $("rx-file");
  const picker = $("rx-file-picker");
  const domain = $("rx-input-domain");
  const defaults = state.rxDefaults;
  fileInput.disabled = state.rxBusy || defaultMode;
  picker.classList.toggle("is-default", defaultMode);
  if (defaultMode) {
    fileInput.value = "";
    $("rx-file-name").textContent = defaults.input_name || "内置默认采集";
    $("rx-file-meta").textContent = `MATLAB HDF5 · ${defaults.input_domain === "frequency" ? "频域" : "时域"} · 本地内置夹具`;
    domain.value = defaults.input_domain || "frequency";
    domain.disabled = true;
    $("rx-noise-variance").value = String(defaults.noise_variance ?? 0);
    $("rx-submit-hint").textContent = `将使用内置采集 ${defaults.input_name || "默认夹具"} 开始解码`;
    return;
  }
  const file = fileInput.files[0];
  if (!file) {
    $("rx-file-name").textContent = "选择 .h5、.hdf5 或 .npz 文件";
    $("rx-file-meta").textContent = "本地处理 · 单文件上限 12 MiB";
    domain.disabled = false;
    domain.value = "time";
  }
  $("rx-submit-hint").textContent = file ? "已选择输入文件，可以开始解码" : "选择输入文件后开始解码";
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("load", () => {
      const result = String(reader.result || "");
      const comma = result.indexOf(",");
      if (comma < 0) reject(new Error("无法编码输入文件。"));
      else resolve(result.slice(comma + 1));
    });
    reader.addEventListener("error", () => reject(new Error("读取输入文件失败。")));
    reader.readAsDataURL(file);
  });
}

function setRxBusy(busy) {
  state.rxBusy = busy;
  const button = $("rx-decode-button");
  button.disabled = busy;
  $("rx-file").disabled = busy || isUsingDefaultRxCapture();
  $("rx-use-default").disabled = busy || !state.rxDefaults.available;
  $("rx-config-select").disabled = busy;
  $("rx-decode-label").textContent = busy ? "正在分析…" : "开始接收分析";
  $("rx-submit-hint").textContent = busy
    ? (isUsingDefaultRxCapture() ? "内置采集正在解码，可能需要数秒" : "文件正在上传并解码，较大波形可能需要数秒")
    : (isUsingDefaultRxCapture() ? `将使用内置采集 ${state.rxDefaults.input_name || "默认夹具"} 开始解码` : "选择输入文件后开始解码");
  $("rx-status-pill").textContent = busy ? "分析中" : state.rxResult ? "已完成" : "等待输入";
  $("rx-status-pill").className = `status-pill ${busy ? "running" : state.rxResult ? "completed" : "idle"}`;
}

function displayMetadata(input) {
  const list = $("rx-metadata");
  list.replaceChildren();
  const entries = input && typeof input === "object" ? Object.entries(input) : [];
  if (!entries.length) entries.push(["输入", "无元数据"]);
  entries.forEach(([key, value]) => {
    const term = document.createElement("dt");
    const description = document.createElement("dd");
    term.textContent = String(key).replaceAll("_", " ");
    if (value === null || value === undefined) description.textContent = "—";
    else if (typeof value === "object") description.textContent = JSON.stringify(value);
    else description.textContent = String(value);
    list.append(term, description);
  });
}

function renderRxResult(result) {
  if (!result || typeof result !== "object" || typeof result.decode_id !== "string") {
    throw new Error("接收分析接口没有返回译码记录编号。请确认网页和后端来自同一版本后重试。");
  }
  state.rxResult = result;
  $("rx-results-empty").classList.add("hidden");
  $("rx-results-content").classList.remove("hidden");
  $("rx-decode-id").textContent = result.decode_id || "—";
  $("rx-result-detector").textContent = result.detector || "—";
  const statuses = Array.isArray(result.crc_status) ? result.crc_status : [];
  const passed = Number.isFinite(Number(result.crc_pass_count)) ? Number(result.crc_pass_count) : statuses.filter(Boolean).length;
  const total = Number.isFinite(Number(result.block_count)) ? Number(result.block_count) : statuses.length;
  $("rx-crc-summary").textContent = `${passed} / ${total}`;
  $("rx-bits-shape").textContent = Array.isArray(result.bits_shape) ? result.bits_shape.join(" × ") : "—";
  displayMetadata(result.input);

  const body = $("rx-crc-body");
  body.replaceChildren();
  statuses.forEach((status, index) => {
    const row = document.createElement("tr");
    const label = document.createElement("td");
    const value = document.createElement("td");
    const badge = document.createElement("span");
    label.textContent = Array.isArray(result.crc_labels) && result.crc_labels[index]
      ? result.crc_labels[index] : `UE${index}`;
    badge.className = `crc-badge ${status ? "pass" : "fail"}`;
    badge.textContent = status ? "通过" : "失败";
    value.append(badge);
    row.append(label, value);
    body.append(row);
  });
  if (!statuses.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 2;
    cell.className = "muted";
    cell.textContent = "响应中没有逐块 CRC 数据。";
    row.append(cell);
    body.append(row);
  }
  $("rx-crc-count").textContent = `${statuses.length} 条记录`;
  for (const [id, url] of [["rx-download-npz", result.output_npz_url], ["rx-download-json", result.output_json_url]]) {
    const link = $(id);
    link.classList.toggle("hidden", !url);
    if (url) link.href = url;
  }
  const users = Array.isArray(result.constellation?.users) ? result.constellation.users : [];
  const count = users.reduce((total, user) => total + Math.min(user.real?.length || 0, user.imag?.length || 0), 0);
  $("rx-constellation-count").textContent = `${count} 个复数符号 · 按 UE 着色`;
  requestAnimationFrame(() => drawConstellation(result.constellation));
}

function drawConstellation(constellation) {
  const canvas = $("rx-constellation");
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const users = Array.isArray(constellation?.users) ? constellation.users : [];
  const series = users.length ? users : [{
    name: "接收流",
    real: Array.isArray(constellation?.real) ? constellation.real : [],
    imag: Array.isArray(constellation?.imag) ? constellation.imag : [],
  }];
  let maxAbs = 0;
  const plotted = series.map((user) => {
    const count = Math.min(user.real?.length || 0, user.imag?.length || 0);
    const step = Math.max(1, Math.ceil(count / 6000));
    const points = [];
    for (let i = 0; i < count; i += step) {
      const x = Number(user.real[i]), y = Number(user.imag[i]);
      if (Number.isFinite(x) && Number.isFinite(y)) {
        points.push([x, y]);
        maxAbs = Math.max(maxAbs, Math.abs(x), Math.abs(y));
      }
    }
    return { name: user.name || "接收流", points };
  }).filter((user) => user.points.length);
  const width = rect.width, height = rect.height;
  const pad = 32;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfcfa";
  ctx.fillRect(0, 0, width, height);
  if (!plotted.length) {
    ctx.fillStyle = "#74807b";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("没有可绘制的星座点", width / 2, height / 2);
    return;
  }
  maxAbs = Math.max(1e-9, maxAbs) * 1.08;
  const px = (value) => pad + ((value + maxAbs) / (2 * maxAbs)) * (width - pad * 2);
  const py = (value) => height - pad - ((value + maxAbs) / (2 * maxAbs)) * (height - pad * 2);
  ctx.strokeStyle = "#e1e7e3";
  ctx.lineWidth = 1;
  for (let i = -2; i <= 2; i++) {
    const value = maxAbs * i / 2;
    ctx.beginPath(); ctx.moveTo(px(value), pad); ctx.lineTo(px(value), height - pad); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad, py(value)); ctx.lineTo(width - pad, py(value)); ctx.stroke();
  }
  ctx.strokeStyle = "#9ba7a1";
  ctx.beginPath(); ctx.moveTo(px(0), pad); ctx.lineTo(px(0), height - pad); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad, py(0)); ctx.lineTo(width - pad, py(0)); ctx.stroke();
  const legend = document.createElement("span");
  legend.className = "rx-constellation-legend";
  plotted.forEach((user, index) => {
    const color = detectorColors[index % detectorColors.length];
    ctx.fillStyle = `${color}88`;
    user.points.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(px(x), py(y), 1.7, 0, Math.PI * 2); ctx.fill(); });
    const item = document.createElement("i");
    item.style.setProperty("--legend-color", color);
    item.textContent = user.name;
    legend.append(item);
  });
  const chartTitle = canvas.closest(".chart-card")?.querySelector(".chart-title");
  if (chartTitle) {
    chartTitle.querySelector(".rx-constellation-legend")?.remove();
    chartTitle.append(legend);
  }
  ctx.fillStyle = "#66736e";
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText("I", width - pad + 8, py(0) - 6);
  ctx.fillText("Q", px(0) + 7, pad - 8);
}

async function decodeRx(event) {
  event.preventDefault();
  if (state.rxBusy) return;
  const useDefaultCapture = isUsingDefaultRxCapture();
  const file = $("rx-file").files[0];
  const inputFormat = selectedInputFormat(file);
  if (!useDefaultCapture && (!file || !inputFormat)) return showAlert("请选择 .h5、.hdf5 或 .npz 输入文件。", true);
  const configName = $("rx-config-select").value;
  const configText = $("rx-config-editor").value.trim();
  if (!configName && !configText) return showAlert("请选择已有 RX 配置，或在右侧粘贴完整 TOML 配置。", true);
  const noiseVariance = Number($("rx-noise-variance").value);
  if (!Number.isFinite(noiseVariance) || noiseVariance < 0) return showAlert("噪声方差必须是非负数。", true);
  setRxBusy(true);
  try {
    const payload = {
      use_default_capture: useDefaultCapture,
      config_name: configName,
      config_text: configText,
      input_name: useDefaultCapture ? state.rxDefaults.input_name : file.name,
      input_format: useDefaultCapture ? (state.rxDefaults.input_format || "matlab-h5") : inputFormat,
      input_domain: useDefaultCapture ? (state.rxDefaults.input_domain || "frequency") : $("rx-input-domain").value,
      noise_variance: noiseVariance,
      channel_estimator: useDefaultCapture ? (state.rxDefaults.channel_estimator || "dmrs") : "dmrs",
      detector: $("rx-detector").value,
      device: $("rx-device").value,
    };
    if (useDefaultCapture && Number.isFinite(Number(state.rxDefaults.max_delay_spread_s))) {
      payload.max_delay_spread_s = Number(state.rxDefaults.max_delay_spread_s);
    }
    if (!useDefaultCapture) payload.input_base64 = await fileToBase64(file);
    const parameter = $("rx-detector-parameter").value.trim();
    const damping = $("rx-detector-damping").value.trim();
    if (parameter) payload.detector_parameter = Number(parameter);
    if (damping) payload.detector_damping = Number(damping);
    const result = await api("/api/rx/decode", { method: "POST", body: JSON.stringify(payload) });
    renderRxResult(result);
  } catch (error) {
    showAlert(error.message);
    setRxBusy(false);
    $("rx-status-pill").textContent = "失败";
    $("rx-status-pill").className = "status-pill failed";
    return;
  } finally {
    if (state.rxBusy) setRxBusy(false);
  }
}

function renderResults(allPoints) {
  const skippedCount = allPoints.filter((point) => point && point.skipped).length;
  const points = allPoints.filter((point) => point && !point.skipped);
  const hasPoints = points.length > 0;
  $("results-empty").classList.toggle("hidden", hasPoints);
  $("results-content").classList.toggle("hidden", !hasPoints);
  $("result-subtitle").textContent = hasPoints
    ? `${new Set(points.map((point) => point.detector)).size} 个检测器 · ${points.length} 个测量点${skippedCount ? ` · 跳过 ${skippedCount} 个点` : ""}`
    : "完成仿真后将在此显示结果";
  if (!hasPoints) return;
  const body = $("metrics-body");
  body.replaceChildren();
  [...points].sort((a, b) => String(a.detector).localeCompare(String(b.detector)) || Number(a.snr_db) - Number(b.snr_db)).forEach((point) => {
    const row = document.createElement("tr");
    const values = [point.detector ?? "—", formatMetric(point.snr_db, 2), formatMetric(point.bler), formatMetric(point.ber), formatMetric(point.crc_fail_rate), point.frames ?? "—", point.transport_blocks ?? "—", point.block_errors ?? "—", Number.isFinite(Number(point.runtime_s)) ? `${formatMetric(point.runtime_s, 2)} s` : "—"];
    values.forEach((value) => { const cell = document.createElement("td"); cell.textContent = String(value); row.append(cell); });
    body.append(row);
  });
  $("table-count").textContent = `${points.length} 条记录`;
  requestAnimationFrame(() => drawChart(points));
}

function drawChart(points) {
  const canvas = $("bler-chart");
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const width = rect.width, height = rect.height;
  const pad = { left: 62, right: 22, top: 15, bottom: 46 };
  const plotW = width - pad.left - pad.right, plotH = height - pad.top - pad.bottom;
  const valid = points.filter((p) => Number.isFinite(Number(p.snr_db)) && Number.isFinite(Number(p.bler)) && Number(p.bler) >= 0);
  if (!valid.length) return;
  const plotBler = (point) => Number(point.bler) > 0
    ? Number(point.bler) : 1 / Math.max(1, Number(point.transport_blocks) || 1000);
  const snrs = valid.map((p) => Number(p.snr_db));
  let xMin = Math.min(...snrs), xMax = Math.max(...snrs);
  if (xMin === xMax) { xMin -= 1; xMax += 1; }
  const yMinLog = Math.floor(Math.log10(Math.min(...valid.map(plotBler), .001)));
  const yMaxLog = 0;
  const x = (v) => pad.left + ((v - xMin) / (xMax - xMin)) * plotW;
  const y = (v) => pad.top + ((yMaxLog - Math.log10(Math.max(v, 10 ** yMinLog))) / (yMaxLog - yMinLog || 1)) * plotH;
  ctx.clearRect(0, 0, width, height);
  ctx.font = "10px ui-monospace, monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let power = yMaxLog; power >= yMinLog; power--) {
    const py = y(10 ** power);
    ctx.strokeStyle = "#e3e8e4"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(pad.left, py); ctx.lineTo(width - pad.right, py); ctx.stroke();
    ctx.fillStyle = "#74807b"; ctx.fillText(power === 0 ? "1" : `10^${power}`, pad.left - 10, py);
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  const ticks = Math.min(8, Math.max(2, Math.round(plotW / 90)));
  for (let i = 0; i <= ticks; i++) {
    const value = xMin + ((xMax - xMin) * i) / ticks;
    const px = x(value);
    ctx.strokeStyle = "#edf0ed"; ctx.beginPath(); ctx.moveTo(px, pad.top); ctx.lineTo(px, height - pad.bottom); ctx.stroke();
    ctx.fillStyle = "#74807b"; ctx.fillText(formatMetric(value, 1), px, height - pad.bottom + 10);
  }
  ctx.fillStyle = "#52605a"; ctx.font = "11px sans-serif"; ctx.fillText("SNR (dB)", pad.left + plotW / 2, height - 16);
  ctx.save(); ctx.translate(15, pad.top + plotH / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.fillText("BLER", 0, 0); ctx.restore();
  const groups = new Map();
  valid.forEach((point) => { const key = String(point.detector ?? "未知检测器"); if (!groups.has(key)) groups.set(key, []); groups.get(key).push(point); });
  const legend = $("chart-legend"); legend.replaceChildren();
  [...groups.entries()].forEach(([name, values], index) => {
    const color = detectorColors[index % detectorColors.length];
    values.sort((a, b) => Number(a.snr_db) - Number(b.snr_db));
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2.2; ctx.beginPath();
    values.forEach((point, i) => { const px = x(Number(point.snr_db)), py = y(plotBler(point)); if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); }); ctx.stroke();
    values.forEach((point) => { const px = x(Number(point.snr_db)), py = y(plotBler(point)); ctx.beginPath(); ctx.arc(px, py, 3.5, 0, Math.PI * 2); ctx.fillStyle = Number(point.bler) === 0 ? "#fffefa" : color; ctx.fill(); ctx.strokeStyle = color; ctx.lineWidth = 1.4; ctx.stroke(); });
    const item = document.createElement("span"); const swatch = document.createElement("i"); swatch.style.background = color; const text = document.createElement("b"); text.textContent = name; item.append(swatch, text); legend.append(item);
  });
}

document.querySelectorAll(".config-tab").forEach((tab) => tab.addEventListener("click", () => switchKind(tab.dataset.kind).catch((error) => showAlert(error.message))));
$("bler-workflow-tab").addEventListener("click", () => switchWorkflow("bler"));
$("rx-workflow-tab").addEventListener("click", () => switchWorkflow("rx"));
$("rx-form").addEventListener("submit", decodeRx);
$("rx-config-select").addEventListener("change", (event) => loadRxConfig(event.target.value).catch((error) => showAlert(error.message)));
$("rx-clear-config").addEventListener("click", () => { $("rx-config-editor").value = ""; $("rx-config-editor").focus(); });
$("rx-use-default").addEventListener("change", (event) => setRxInputMode(event.target.checked));
$("rx-file").addEventListener("change", (event) => {
  const file = event.target.files[0];
  const format = selectedInputFormat(file);
  if (file && isUsingDefaultRxCapture()) {
    $("rx-use-default").checked = false;
    setRxInputMode(false);
  }
  if (!file) {
    setRxInputMode(false);
    return;
  }
  $("rx-file-name").textContent = file.name;
  $("rx-file-meta").textContent = `${format === "matlab-h5" ? "MATLAB HDF5" : format.toUpperCase()} · ${formatFileSize(file.size)}`;
  if (format === "matlab-h5") {
    $("rx-input-domain").value = "frequency";
    $("rx-input-domain").disabled = true;
  } else {
    $("rx-input-domain").disabled = false;
    $("rx-input-domain").value = "time";
  }
  $("rx-submit-hint").textContent = "已选择输入文件，可以开始解码";
});
$("rx-detector").addEventListener("change", (event) => {
  const method = event.target.value;
  const labels = { "k-best": "K 候选数（可选）", ep: "迭代次数（可选）", "mmse-pic": "迭代次数（可选）" };
  const parameter = $("rx-detector-parameter");
  parameter.disabled = !Object.hasOwn(labels, method);
  $("rx-parameter-label").textContent = labels[method] || "该检测器使用默认参数";
  const supportsDamping = method === "ep" || method === "mmse-pic";
  $("rx-detector-damping").disabled = !supportsDamping;
});
$("rx-detector").dispatchEvent(new Event("change"));
$("profile-select").addEventListener("change", async (event) => {
  const kind = state.activeKind;
  if (editor.value !== state.savedTexts[kind] && !window.confirm("当前配置尚未保存，切换文件会丢失修改。确定继续吗？")) {
    event.target.value = state.selected[kind];
    return;
  }
  state.texts[kind] = editor.value;
  state.selected[kind] = event.target.value;
  try { await loadConfig(kind, event.target.value, true); } catch (error) { showAlert(error.message); }
});
editor.addEventListener("input", () => { state.texts[state.activeKind] = editor.value; updateLines(); updateDirty(); });
editor.addEventListener("scroll", () => { $("line-numbers").scrollTop = editor.scrollTop; });
editor.addEventListener("keydown", (event) => {
  if (event.key === "Tab") { event.preventDefault(); const start = editor.selectionStart, end = editor.selectionEnd; editor.setRangeText("  ", start, end, "end"); editor.dispatchEvent(new Event("input")); }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") { event.preventDefault(); saveCurrentConfig(); }
});
$("save-button").addEventListener("click", saveCurrentConfig);
$("reload-button").addEventListener("click", () => {
  if (editor.value !== state.savedTexts[state.activeKind] && !window.confirm("重新载入会丢失未保存的修改。确定继续吗？")) return;
  loadConfig(state.activeKind, state.selected[state.activeKind], true).catch((error) => showAlert(error.message));
});
$("run-button").addEventListener("click", startRun);
$("cancel-button").addEventListener("click", cancelRun);
$("refresh-history").addEventListener("click", loadRuns);
$("alert-close").addEventListener("click", () => $("global-alert").classList.add("hidden"));
$("log-toggle").addEventListener("click", () => { const hidden = $("log-view").classList.toggle("hidden"); $("log-toggle").textContent = hidden ? "展开" : "收起"; });
window.addEventListener("resize", () => {
  if (state.currentJob && state.currentJob.points?.length) drawChart(state.currentJob.points);
  if (state.rxResult) drawConstellation(state.rxResult.constellation);
});

Promise.all([initConfigs(), loadRuns(), initRxConfigs()]).catch((error) => showAlert(error.message));

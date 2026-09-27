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
};

const $ = (id) => document.getElementById(id);
const editor = $("config-editor");
const statusLabels = { queued: "排队中", running: "运行中", completed: "已完成", failed: "失败", cancelled: "已取消" };
const detectorColors = ["#315fdf", "#e97832", "#8b55d9", "#3b9b6c", "#d4a719", "#d34e70"];

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
  const pointCount = Array.isArray(job.points) ? job.points.length : 0;
  const totalPoints = Number(job.total_points) || 0;
  $("job-points").textContent = totalPoints ? `${pointCount} / ${totalPoints}` : String(pointCount);
  $("progress-bar").style.width = totalPoints
    ? `${Math.min(100, pointCount / totalPoints * 100)}%`
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
  renderResults(Array.isArray(job.points) ? job.points : []);
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

function renderResults(points) {
  const hasPoints = points.length > 0;
  $("results-empty").classList.toggle("hidden", hasPoints);
  $("results-content").classList.toggle("hidden", !hasPoints);
  $("result-subtitle").textContent = hasPoints ? `${new Set(points.map((point) => point.detector)).size} 个检测器 · ${points.length} 个测量点` : "完成仿真后将在此显示结果";
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
window.addEventListener("resize", () => { if (state.currentJob && state.currentJob.points?.length) drawChart(state.currentJob.points); });

Promise.all([initConfigs(), loadRuns()]).catch((error) => showAlert(error.message));

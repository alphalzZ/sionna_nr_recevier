"use strict";

const state = {
  configs: { tx: [], channel: [], simulation: [] },
  selected: { tx: "", channel: "", simulation: "" },
  texts: { tx: "", channel: "", simulation: "" },
  savedTexts: { tx: "", channel: "", simulation: "" },
  selectedScenario: "",
  profileCompatibility: { tx: null, channel: null, simulation: null },
  validation: null,
  validationTimer: null,
  validationSerial: 0,
  runSubmitting: false,
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
const seriesColors = ["#315fdf", "#e97832", "#8b55d9", "#3b9b6c", "#d4a719", "#d34e70"];

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
  const kind = state.activeKind;
  const names = state.configs[kind] || [];
  const availability = state.profileCompatibility[kind] || {};
  if (!names.length) {
    const option = document.createElement("option");
    option.textContent = "无可用配置";
    option.value = "";
    select.append(option);
    select.disabled = true;
    return;
  }
  names.forEach((name) => {
    const option = document.createElement("option");
    const result = availability[name];
    option.value = name;
    option.textContent = result?.compatible === false ? `${name} · 不兼容` : name;
    option.title = result?.errors?.map((error) => error.message).join("；") || "";
    option.disabled = result?.compatible === false && name !== state.selected[kind];
    select.append(option);
  });
  select.disabled = false;
  select.value = state.selected[kind] || names[0];
}

function currentRunPayload() {
  return {
    scenario_id: state.selectedScenario,
    tx_name: state.selected.tx,
    channel_name: state.selected.channel,
    simulation_name: state.selected.simulation,
    tx_text: state.texts.tx,
    channel_text: state.texts.channel,
    simulation_text: state.texts.simulation,
  };
}

function populateScenarioSelect() {
  const select = $("scenario-select");
  select.replaceChildren();
  const custom = document.createElement("option");
  custom.value = "";
  custom.textContent = "高级自定义组合";
  select.append(custom);
  state.scenarios.forEach((scenario) => {
    const option = document.createElement("option");
    option.value = scenario.id;
    option.textContent = scenario.label;
    select.append(option);
  });
  select.value = state.selectedScenario;
}

function renderCompatibility(result) {
  const status = $("compatibility-status");
  const button = $("run-button");
  status.replaceChildren();
  if (!result) {
    status.className = "compatibility-status pending";
    status.textContent = "正在检查 TX、信道、仿真配置组合…";
    button.disabled = true;
    return;
  }
  const errors = Array.isArray(result.errors) ? result.errors : [];
  status.className = `compatibility-status ${result.valid ? "valid" : "invalid"}`;
  const heading = document.createElement("strong");
  heading.textContent = result.valid ? "兼容性检查通过" : "当前组合不可运行";
  status.append(heading);
  if (errors.length) {
    const list = document.createElement("ul");
    const kindNames = {
      tx: "发射机",
      channel: "信道",
      simulation: "仿真",
      scenario: "场景",
    };
    errors.forEach((error) => {
      const item = document.createElement("li");
      item.textContent = `${kindNames[error.kind] || error.kind}：${error.message}`;
      list.append(item);
    });
    status.append(list);
  }
  button.disabled = !result.valid || state.runSubmitting;
}

async function validateCurrentRun() {
  const serial = ++state.validationSerial;
  state.validation = null;
  renderCompatibility(null);
  const result = await api("/api/validate-run", {
    method: "POST",
    body: JSON.stringify(currentRunPayload()),
  });
  if (serial !== state.validationSerial) return null;
  state.validation = result;
  renderCompatibility(result);
  return result;
}

async function refreshProfileCompatibility() {
  if (!$("advanced-config").open) return;
  const kind = state.activeKind;
  const result = await api("/api/profile-compatibility", {
    method: "POST",
    body: JSON.stringify({ ...currentRunPayload(), scenario_id: "", kind }),
  });
  state.profileCompatibility[kind] = result.profiles || {};
  populateProfileSelect();
}

function scheduleValidation(refreshProfiles = false) {
  clearTimeout(state.validationTimer);
  state.validationTimer = setTimeout(async () => {
    try {
      await validateCurrentRun();
      if (refreshProfiles) await refreshProfileCompatibility();
    } catch (error) {
      state.validation = null;
      renderCompatibility({
        valid: false,
        errors: [{ kind: "validation", message: error.message }],
      });
    }
  }, 250);
}

function hasUnsavedConfigChanges() {
  return Object.keys(state.texts).some((kind) => state.texts[kind] !== state.savedTexts[kind]);
}

function profileBundleLabel(profiles) {
  return `TX ${profiles.tx || "未选择"} · 信道 ${profiles.channel || "未选择"} · 仿真 ${profiles.simulation || "未选择"}`;
}

function useCustomCombination() {
  state.selectedScenario = "";
  populateScenarioSelect();
  $("scenario-description").textContent = `高级自定义组合；${profileBundleLabel({
    ...state.selected,
  })}。更改任何配置后都会重新检查兼容性。`;
}

async function applyScenario(scenarioId) {
  const scenario = state.scenarios.find((item) => item.id === scenarioId);
  if (!scenario) return;
  if (hasUnsavedConfigChanges() && !window.confirm("切换场景会丢弃未保存的 TOML 修改。继续吗？")) {
    $("scenario-select").value = state.selectedScenario;
    return;
  }
  state.selectedScenario = scenario.id;
  state.selected = {
    tx: scenario.profiles.tx,
    channel: scenario.profiles.channel,
    simulation: scenario.profiles.simulation,
  };
  state.profileCompatibility = { tx: null, channel: null, simulation: null };
  $("scenario-description").textContent = `${scenario.description} 配置：${profileBundleLabel(scenario.profiles)}`;
  populateScenarioSelect();
  populateProfileSelect();
  const loads = ["tx", "channel", "simulation"].map((kind) =>
    loadConfig(kind, state.selected[kind], true),
  );
  await Promise.all(loads);
  editor.value = state.texts[state.activeKind] || "";
  updateLines();
  updateDirty();
  await validateCurrentRun();
  if ($("advanced-config").open) await refreshProfileCompatibility();
}

async function initConfigs() {
  const [configs, scenarioPayload] = await Promise.all([
    api("/api/configs"),
    api("/api/scenarios"),
  ]);
  ["tx", "channel", "simulation"].forEach((kind) => {
    state.configs[kind] = Array.isArray(configs[kind]) ? configs[kind] : [];
  });
  state.scenarios = Array.isArray(scenarioPayload.scenarios) ? scenarioPayload.scenarios : [];
  if (!state.scenarios.length) throw new Error("没有可用的兼容场景；请检查 configs/scenarios.toml");
  state.selectedScenario = state.scenarios[0].id;
  populateScenarioSelect();
  $("run-button").disabled = true;
  await applyScenario(state.selectedScenario);
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
  await validateCurrentRun();
  await refreshProfileCompatibility();
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
    await validateCurrentRun();
    await refreshProfileCompatibility();
  } catch (error) { showAlert(error.message); }
  finally { button.disabled = false; }
}

async function startRun() {
  if (state.runSubmitting) return;
  state.texts[state.activeKind] = editor.value;
  const button = $("run-button");
  state.runSubmitting = true;
  button.disabled = true;
  try {
    const validation = await validateCurrentRun();
    if (!validation?.valid) {
      showAlert("配置组合未通过预检；请先按提示修正。", true);
      return;
    }
    const job = await api("/api/runs", {
      method: "POST",
      body: JSON.stringify(currentRunPayload()),
    });
    state.currentJob = job;
    renderJob(job);
    await loadRuns();
    schedulePoll(250);
  } catch (error) { showAlert(error.message); }
  finally {
    state.runSubmitting = false;
    button.disabled = !state.validation?.valid;
  }
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
  const channelConfigs = state.configs.channel;
  const channelSelect = $("rx-channel-select");
  channelSelect.replaceChildren();
  if (!channelConfigs.length) {
    const empty = document.createElement("option");
    empty.value = "";
    empty.textContent = "无可用信道配置";
    channelSelect.append(empty);
    channelSelect.disabled = true;
  } else {
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "请选择 CDL 配置";
    channelSelect.append(placeholder);
    channelConfigs.forEach((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      channelSelect.append(option);
    });
    channelSelect.disabled = false;
  }
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
  $("rx-max-delay-spread").value = state.rxDefaults.max_delay_spread_s ?? "";
  $("rx-detector").dispatchEvent(new Event("change"));
  $("rx-channel-estimator").value = state.rxDefaults.channel_estimator || "dmrs";
  $("rx-channel-estimator").dispatchEvent(new Event("change"));
  $("rx-default-meta").textContent = state.rxDefaults.available
    ? `${state.rxDefaults.input_name || "本地默认夹具"} · ${state.rxDefaults.config_name || "默认配置"}`
    : "未发现本地默认夹具，请选择上传文件";
  setRxInputMode(defaultToggle.checked);
}

function parseReceiverProfile(text) {
  const profile = {};
  let inReceiver = false;
  String(text || "").split(/\r?\n/).forEach((raw) => {
    const line = raw.trim();
    if (!line || line.startsWith("#")) return;
    if (line.startsWith("[")) { inReceiver = line === "[receiver]"; return; }
    if (!inReceiver) return;
    const match = line.match(/^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$/);
    if (!match) return;
    let value = match[2].split("#")[0].trim().replace(/,$/, "");
    if (/^".*"$/.test(value) || /^'.*'$/.test(value)) value = value.slice(1, -1);
    else if (value === "true" || value === "false") value = value === "true";
    else if (/^-?\d+(\.\d+)?([eE][-+]?\d+)?$/.test(value)) value = Number(value);
    profile[match[1]] = value;
  });
  return profile;
}

// Mirror the RX CLI: the selected profile's [receiver] table seeds the form,
// so the knobs shown are the ones that actually produce the decode.
function applyReceiverProfile(text) {
  const profile = parseReceiverProfile(text);
  const set = (id, value) => {
    if (value !== undefined && value !== null) $(id).value = String(value);
  };
  set("rx-detector", profile.detector);
  if (profile.detector === "soft-mmse-pic") {
    set("rx-detector-parameter", profile.detector_parameter ?? 1);
    set("rx-detector-damping", profile.detector_damping ?? 0.25);
  } else {
    set("rx-detector-parameter", profile.detector_parameter);
    set("rx-detector-damping", profile.detector_damping);
  }
  set("rx-noise-variance", profile.noise_variance);
  set("rx-max-delay-spread", profile.max_delay_spread_s);
  $("rx-detector").dispatchEvent(new Event("change"));
}

async function loadRxConfig(name) {
  if (!name) return;
  const config = await api(`/api/configs/rx/${encodeURIComponent(name)}`);
  $("rx-config-editor").value = config.text || "";
  applyReceiverProfile(config.text);
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
    $("rx-scrambling").value = "";
    $("rx-scrambling").disabled = true;
    $("rx-scrambling-name").textContent = "加扰序列 H5（可选）";
    $("rx-scrambling-meta").textContent = "抓包使用非 RNTI 扰码时上传，否则按 RNTI 推导";
    // The built-in capture only decodes against its own profile, so switch to it
    // instead of silently applying a mismatched MCS to the fixture.
    if (defaults.config_name && $("rx-config-select").value !== defaults.config_name) {
      $("rx-config-select").value = defaults.config_name;
      loadRxConfig(defaults.config_name).catch((error) => showAlert(error.message));
    }
    $("rx-file-name").textContent = defaults.input_name || "内置默认采集";
    $("rx-file-meta").textContent = `MATLAB HDF5 · ${defaults.input_domain === "frequency" ? "频域" : "时域"} · 本地内置夹具`;
    domain.value = defaults.input_domain || "frequency";
    domain.disabled = true;
    $("rx-noise-variance").value = String(defaults.noise_variance ?? 0);
    $("rx-submit-hint").textContent = `将使用内置采集 ${defaults.input_name || "默认夹具"} 开始解码`;
    return;
  }
  $("rx-scrambling").disabled = state.rxBusy;
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

function syncRxChannelControl() {
  const channelSelect = $("rx-channel-select");
  const needsChannel = $("rx-channel-estimator").value === "dmrs-lmmse";
  channelSelect.disabled = state.rxBusy || !needsChannel || !channelSelect.options.length;
  channelSelect.required = needsChannel;
}

function setRxBusy(busy) {
  state.rxBusy = busy;
  const button = $("rx-decode-button");
  button.disabled = busy;
  $("rx-file").disabled = busy || isUsingDefaultRxCapture();
  $("rx-scrambling").disabled = busy || isUsingDefaultRxCapture();
  $("rx-use-default").disabled = busy || !state.rxDefaults.available;
  $("rx-config-select").disabled = busy;
  $("rx-channel-estimator").disabled = busy;
  syncRxChannelControl();
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

function cbCrcCell(entry) {
  const cell = document.createElement("td");
  if (!entry || !Array.isArray(entry.status) || !entry.status.length) {
    cell.className = "muted";
    cell.textContent = "—";
    return cell;
  }
  const total = entry.status.length;
  const passed = entry.status.filter(Boolean).length;
  const summary = document.createElement("span");
  summary.className = `cb-crc-summary ${passed === total ? "pass" : passed ? "partial" : "fail"}`;
  summary.textContent = `${passed}/${total}`;
  const grid = document.createElement("div");
  grid.className = "cb-crc-cells";
  entry.status.forEach((ok, blockIndex) => {
    const block = document.createElement("span");
    block.className = `cb-crc-cell ${ok ? "pass" : "fail"}`;
    block.textContent = String(blockIndex);
    block.title = `码块 ${blockIndex}：${ok ? "通过" : "失败"}`;
    grid.append(block);
  });
  cell.append(summary, grid);
  return cell;
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

  const cbUsers = Array.isArray(result.cb_crc?.users) ? result.cb_crc.users : [];
  const body = $("rx-crc-body");
  body.replaceChildren();
  statuses.forEach((status, index) => {
    const row = document.createElement("tr");
    const label = document.createElement("td");
    const value = document.createElement("td");
    const badge = document.createElement("span");
    const labelText = Array.isArray(result.crc_labels) && result.crc_labels[index]
      ? result.crc_labels[index] : `UE${index}`;
    label.textContent = labelText;
    badge.className = `crc-badge ${status ? "pass" : "fail"}`;
    badge.textContent = status ? "通过" : "失败";
    value.append(badge);
    const entry = cbUsers.length === statuses.length
      ? cbUsers[index]
      : cbUsers.find((user) => user?.name === labelText) || null;
    row.append(label, value, cbCrcCell(entry));
    body.append(row);
  });
  if (!statuses.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 3;
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
  $("rx-constellation-count").textContent = `${count} 个复数符号 · 每 UE/层独立坐标`;
  requestAnimationFrame(() => drawConstellation(result.constellation));
}

function drawConstellation(constellation) {
  const grid = $("rx-constellation-grid");
  if (!grid) return;
  const users = Array.isArray(constellation?.users) ? constellation.users : [];
  const series = users.length ? users : [{
    name: "接收流",
    real: Array.isArray(constellation?.real) ? constellation.real : [],
    imag: Array.isArray(constellation?.imag) ? constellation.imag : [],
  }];
  const prepared = series.map((user) => {
    const count = Math.min(user.real?.length || 0, user.imag?.length || 0);
    const step = Math.max(1, Math.ceil(count / 6000));
    const points = [];
    let maxAbs = 0;
    for (let i = 0; i < count; i += step) {
      const x = Number(user.real[i]), y = Number(user.imag[i]);
      if (Number.isFinite(x) && Number.isFinite(y)) {
        points.push([x, y]);
        maxAbs = Math.max(maxAbs, Math.abs(x), Math.abs(y));
      }
    }
    return { name: user.name || "接收流", points, maxAbs };
  }).filter((user) => user.points.length);
  grid.replaceChildren();
  if (!prepared.length) {
    const empty = document.createElement("p");
    empty.className = "muted constellation-empty";
    empty.textContent = "没有可绘制的星座点";
    grid.append(empty);
    return;
  }
  prepared.forEach((user, index) => {
    const color = seriesColors[index % seriesColors.length];
    const cell = document.createElement("figure");
    cell.className = "constellation-cell";
    const caption = document.createElement("figcaption");
    const swatch = document.createElement("i");
    swatch.style.setProperty("--legend-color", color);
    const name = document.createElement("b");
    name.textContent = user.name;
    const count = document.createElement("small");
    count.textContent = `${user.points.length} 点`;
    caption.append(swatch, name, count);
    const canvas = document.createElement("canvas");
    canvas.className = "constellation-canvas";
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", `${user.name} 星座散点图`);
    cell.append(caption, canvas);
    grid.append(cell);
    paintConstellation(canvas, user, color);
  });
}

function paintConstellation(canvas, user, color) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const width = rect.width, height = rect.height;
  const pad = 26;
  // Each UE gets its own scale, otherwise a strong user flattens the others.
  const maxAbs = Math.max(1e-9, user.maxAbs) * 1.08;
  const px = (value) => pad + ((value + maxAbs) / (2 * maxAbs)) * (width - pad * 2);
  const py = (value) => height - pad - ((value + maxAbs) / (2 * maxAbs)) * (height - pad * 2);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfcfa";
  ctx.fillRect(0, 0, width, height);
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
  ctx.fillStyle = `${color}88`;
  user.points.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(px(x), py(y), 1.5, 0, Math.PI * 2); ctx.fill(); });
  ctx.fillStyle = "#66736e";
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText("I", width - pad + 6, py(0) - 5);
  ctx.fillText("Q", px(0) + 5, pad - 7);
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
      channel_estimator: $("rx-channel-estimator").value,
      detector: $("rx-detector").value,
      device: $("rx-device").value,
    };
    if ($("rx-channel-estimator").value === "dmrs-lmmse") {
      const channelName = $("rx-channel-select").value;
      if (!channelName) return showAlert("DMRS-LMMSE 需要选择匹配采集的信道配置。", true);
      payload.channel_config_name = channelName;
    }
    const delaySpread = $("rx-max-delay-spread").value.trim();
    if (delaySpread) payload.max_delay_spread_s = Number(delaySpread);
    else if (useDefaultCapture && Number.isFinite(Number(state.rxDefaults.max_delay_spread_s))) {
      payload.max_delay_spread_s = Number(state.rxDefaults.max_delay_spread_s);
    }
    if (!useDefaultCapture) payload.input_base64 = await fileToBase64(file);
    const parameter = $("rx-detector-parameter").value.trim();
    const damping = $("rx-detector-damping").value.trim();
    if (parameter) payload.detector_parameter = Number(parameter);
    if (damping) payload.detector_damping = Number(damping);
    const scrambling = $("rx-scrambling").files[0];
    if (scrambling) {
      payload.scrambling_name = scrambling.name;
      payload.scrambling_base64 = await fileToBase64(scrambling);
    }
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

function comboLabel(point) {
  return `${point.channel_estimator || "—"} / ${point.detector || "—"}`;
}

function renderResults(allPoints) {
  const skippedCount = allPoints.filter((point) => point && point.skipped).length;
  const points = allPoints.filter((point) => point && !point.skipped);
  const hasPoints = points.length > 0;
  $("results-empty").classList.toggle("hidden", hasPoints);
  $("results-content").classList.toggle("hidden", !hasPoints);
  $("result-subtitle").textContent = hasPoints
    ? `${new Set(points.map(comboLabel)).size} 个估计器/检测器组合 · ${points.length} 个测量点${skippedCount ? ` · 跳过 ${skippedCount} 个点` : ""}`
    : "完成仿真后将在此显示结果";
  if (!hasPoints) return;
  const body = $("metrics-body");
  body.replaceChildren();
  [...points].sort((a, b) =>
    String(a.channel_estimator).localeCompare(String(b.channel_estimator))
    || String(a.detector).localeCompare(String(b.detector))
    || Number(a.snr_db) - Number(b.snr_db)
  ).forEach((point) => {
    const row = document.createElement("tr");
    const values = [comboLabel(point), formatMetric(point.snr_db, 2), formatMetric(point.bler), formatMetric(point.ber), formatMetric(point.crc_fail_rate), point.frames ?? "—", point.transport_blocks ?? "—", point.block_errors ?? "—", Number.isFinite(Number(point.runtime_s)) ? `${formatMetric(point.runtime_s, 2)} s` : "—"];
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
  valid.forEach((point) => { const key = comboLabel(point); if (!groups.has(key)) groups.set(key, []); groups.get(key).push(point); });
  const legend = $("chart-legend"); legend.replaceChildren();
  [...groups.entries()].forEach(([name, values], index) => {
    const color = seriesColors[index % seriesColors.length];
    values.sort((a, b) => Number(a.snr_db) - Number(b.snr_db));
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2.2;
    ctx.beginPath();
    values.forEach((point, i) => { const px = x(Number(point.snr_db)), py = y(plotBler(point)); if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); });
    ctx.stroke();
    values.forEach((point) => { const px = x(Number(point.snr_db)), py = y(plotBler(point)); ctx.beginPath(); ctx.arc(px, py, 3.5, 0, Math.PI * 2); ctx.fillStyle = Number(point.bler) === 0 ? "#fffefa" : color; ctx.fill(); ctx.strokeStyle = color; ctx.lineWidth = 1.4; ctx.stroke(); });
    const item = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.style.background = color;
    const text = document.createElement("b");
    text.textContent = name;
    item.append(swatch, text);
    legend.append(item);
  });
}

document.querySelectorAll(".config-tab").forEach((tab) => tab.addEventListener("click", () => switchKind(tab.dataset.kind).catch((error) => showAlert(error.message))));
$("bler-workflow-tab").addEventListener("click", () => switchWorkflow("bler"));
$("rx-workflow-tab").addEventListener("click", () => switchWorkflow("rx"));
$("rx-channel-estimator").addEventListener("change", syncRxChannelControl);
$("rx-form").addEventListener("submit", decodeRx);
$("rx-config-select").addEventListener("change", (event) => loadRxConfig(event.target.value).catch((error) => showAlert(error.message)));
$("rx-config-editor").addEventListener("input", (event) => {
  applyReceiverProfile(event.target.value);
});
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
$("rx-scrambling").addEventListener("change", (event) => {
  const file = event.target.files[0];
  $("rx-scrambling-name").textContent = file ? file.name : "加扰序列 H5（可选）";
  $("rx-scrambling-meta").textContent = file
    ? `${formatFileSize(file.size)} · 将覆盖按 RNTI 推导的扰码`
    : "抓包使用非 RNTI 扰码时上传，否则按 RNTI 推导";
});
$("rx-detector").addEventListener("change", (event) => {
  const method = event.target.value;
  const labels = {
    "k-best": "K 候选数（可选）",
    ep: "迭代次数（可选）",
    "mmse-pic": "迭代次数（可选）",
    "soft-mmse-pic": "LDPC 外反馈次数（可选）",
  };
  const parameter = $("rx-detector-parameter");
  parameter.disabled = !Object.hasOwn(labels, method);
  $("rx-parameter-label").textContent = labels[method] || "该检测器使用默认参数";
  const supportsDamping = method === "ep" || method === "mmse-pic" || method === "soft-mmse-pic";
  $("rx-detector-damping").disabled = !supportsDamping;
  if (event.isTrusted && method === "soft-mmse-pic") {
    parameter.value = "1";
    $("rx-detector-damping").value = "0.25";
  }
});
$("rx-detector").dispatchEvent(new Event("change"));
$("scenario-select").addEventListener("change", (event) => {
  if (event.target.value) {
    applyScenario(event.target.value).catch((error) => showAlert(error.message));
  } else if (state.selectedScenario) {
    useCustomCombination();
    scheduleValidation(true);
  }
});
$("advanced-config").addEventListener("toggle", () => {
  if ($("advanced-config").open) refreshProfileCompatibility().catch((error) => showAlert(error.message));
});
$("profile-select").addEventListener("change", async (event) => {
  const kind = state.activeKind;
  if (editor.value !== state.savedTexts[kind] && !window.confirm("当前配置尚未保存，切换文件会丢失修改。确定继续吗？")) {
    event.target.value = state.selected[kind];
    return;
  }
  state.texts[kind] = editor.value;
  state.selected[kind] = event.target.value;
  state.profileCompatibility[kind] = null;
  useCustomCombination();
  try {
    await loadConfig(kind, event.target.value, true);
    await validateCurrentRun();
    await refreshProfileCompatibility();
  } catch (error) { showAlert(error.message); }
});
editor.addEventListener("input", () => {
  state.texts[state.activeKind] = editor.value;
  if (state.selectedScenario) useCustomCombination();
  updateLines();
  updateDirty();
  scheduleValidation();
});
editor.addEventListener("blur", () => {
  if ($("advanced-config").open) refreshProfileCompatibility().catch((error) => showAlert(error.message));
});
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

initConfigs()
  .then(() => Promise.all([loadRuns(), initRxConfigs()]))
  .catch((error) => showAlert(error.message));

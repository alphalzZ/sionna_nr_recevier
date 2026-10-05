"use strict";

const state = {
  configs: { tx: [], channel: [], simulation: [], rt: [] },
  selected: { tx: "", channel: "", simulation: "" },
  texts: { tx: "", channel: "", simulation: "" },
  savedTexts: { tx: "", channel: "", simulation: "" },
  selectedScenario: "",
  scenarios: [],
  profileCompatibility: { tx: null, channel: null, simulation: null },
  validation: null,
  validationTimer: null,
  validationSerial: 0,
  runSubmitting: false,
  activeKind: "tx",
  backend: "cdl",
  currentJob: null,
  runs: [],
  pollTimer: null,
  rtInitialized: false,
  rtOptions: null,
  rtOptionsError: "",
  rtScenarios: [],
  rtPresetId: "",
  rtScenarioId: "",
  rtSelected: { tx: "", rt: "", simulation: "" },
  rtSettings: null,
  rtDraftSettings: null,
  rtText: "",
  rtTxText: "",
  rtSimulationText: "",
  rtSceneId: null,
  rtSceneFilename: "",
  rtSceneBundleHash: "",
  rtMeshSummary: null,
  rtWaveform: "dft",
  rtBudget: "quick",
  rtDirty: false,
  rtExpertDirty: false,
  rtConfigTimer: null,
  rtConfigSerial: 0,
  rtConfigPending: false,
  rtConfigError: "",
  rtConfigPromise: null,
  rtUploadPending: false,
  rtLimitAlertText: "",
  rtResultUser: "all",
  resultBackend: "cdl",
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

function isRtScenario(scenario) {
  return scenario?.channel_backend === "rt" || Boolean(scenario?.profiles?.rt);
}

function currentRunPayload() {
  if (state.backend === "rt") {
    const payload = {
      channel_backend: "rt",
      tx_name: state.rtSelected.tx,
      rt_name: state.rtSelected.rt,
      simulation_name: state.rtSelected.simulation,
      tx_text: state.rtTxText,
      rt_text: state.rtText,
      simulation_text: state.rtSimulationText,
    };
    if (state.rtScenarioId) payload.scenario_id = state.rtScenarioId;
    if (state.rtSceneId) payload.scene_id = state.rtSceneId;
    return payload;
  }
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
  state.scenarios.filter((scenario) => !isRtScenario(scenario)).forEach((scenario) => {
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
    status.textContent = state.backend === "rt" ? "正在检查 RT、TX、仿真配置与场景资源…" : "正在检查 TX、信道、仿真配置组合…";
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
      const kindNames = {
        tx: "发射机",
        channel: "信道",
        simulation: "仿真",
        scenario: "场景",
        rt: "RT",
        scene: "场景",
      };
      const location = error.field ? `${error.field}：` : "";
      item.textContent = `${kindNames[error.kind] || error.kind}：${location}${error.message}`;
      list.append(item);
    });
    status.append(list);
  }
  button.disabled = !result.valid || state.runSubmitting
    || (state.backend === "rt" && (state.rtConfigPending || Boolean(state.rtConfigError) || state.rtExpertDirty || state.rtUploadPending));
  updateRtLimitAlert(result);
}

function updateRtLimitAlert(result) {
  const previous = state.rtLimitAlertText;
  const limitErrors = state.backend === "rt" && Array.isArray(result?.errors)
    ? result.errors.filter((error) =>
      typeof error?.message === "string"
      && error.message.includes("RT Web limits exceeded:"))
    : [];
  if (!limitErrors.length) {
    if (previous && $("alert-text").textContent === previous) {
      $("global-alert").classList.add("hidden");
    }
    state.rtLimitAlertText = "";
    return;
  }
  const details = limitErrors.map((error) => error.message).join("；");
  const message = `RT 配置超出 Web 限制：${details}`;
  if (message !== previous) {
    showAlert(message);
    state.rtLimitAlertText = message;
  }
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
  const scenario = state.scenarios.find((item) => item.id === scenarioId && !isRtScenario(item));
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
  ["tx", "channel", "simulation", "rt"].forEach((kind) => {
    state.configs[kind] = Array.isArray(configs[kind]) ? configs[kind] : [];
  });
  state.scenarios = Array.isArray(scenarioPayload.scenarios) ? scenarioPayload.scenarios : [];
  const cdlScenarios = state.scenarios.filter((scenario) => !isRtScenario(scenario));
  if (!cdlScenarios.length) throw new Error("没有可用的 CDL 兼容场景；请检查 configs/scenarios.toml");
  state.selectedScenario = cdlScenarios[0].id;
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
  if (state.backend === "cdl") state.texts[state.activeKind] = editor.value;
  const button = $("run-button");
  state.runSubmitting = true;
  button.disabled = true;
  try {
    if (state.backend === "rt") await flushRtSettings();
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
    if (state.validation) renderCompatibility(state.validation);
  }
}

function runDownloadUrl(job, artifactField, basename) {
  const available = artifactField === "output_validation"
    ? job?.output_validation || job?.validation_url
    : job?.[artifactField];
  if (!job?.id || !basename || !available) return "";
  return `/api/runs/${encodeURIComponent(job.id)}/${basename}`;
}

function requestRtValidationReport(job) {
  const id = job?.id;
  if (!id || !runDownloadUrl(job, "output_validation", "validation.json")
      || state.rtValidationReportRequestedJobId === id
      || state.rtValidationReportJobId === id) return;
  state.rtValidationReportRequestedJobId = id;
  const serial = state.rtValidationReportSerial;
  api(runDownloadUrl(job, "output_validation", "validation.json"))
    .then((report) => {
      if (serial !== state.rtValidationReportSerial || state.rtValidationSelectedJobId !== id) return;
      state.rtValidationReport = report;
      state.rtValidationReportJobId = id;
      state.rtValidationError = "";
      renderRtValidationInfo(state.currentJob);
    })
    .catch((error) => {
      if (serial !== state.rtValidationReportSerial || state.rtValidationSelectedJobId !== id) return;
      state.rtValidationError = error.message;
      renderRtValidationInfo(state.currentJob);
    });
}

function renderRtValidationInfo(job) {
  const panel = $("rt-result-summary");
  const isRt = job.channel_backend === "rt" || job.backend === "rt" || Boolean(job.config_names?.rt);
  panel.classList.toggle("hidden", !isRt);
  $("rt-user-filter-wrap").classList.toggle("hidden", !isRt);
  if (state.rtValidationSelectedJobId !== job.id) {
    state.rtValidationSelectedJobId = job.id;
    state.rtValidationReportSerial = (state.rtValidationReportSerial || 0) + 1;
    state.rtValidationReportRequestedJobId = null;
    state.rtValidationReportJobId = null;
    state.rtValidationReport = null;
    state.rtValidationError = "";
  }
  if (!isRt) return;
  requestRtValidationReport(job);
  const inlineReport = job.validation_summary || job.web_validation || job.validation || job.metadata?.web_validation || job.manifest?.web_validation || {};
  const report = state.rtValidationReportJobId === job.id && state.rtValidationReport
    ? state.rtValidationReport : inlineReport;
  const details = $("rt-validation-details");
  details.replaceChildren();
  const add = (label, value) => {
    if (value === undefined || value === null || value === "") return;
    const item = document.createElement("div");
    const key = document.createElement("span");
    key.textContent = label;
    const text = document.createElement("strong");
    text.textContent = typeof value === "object" ? JSON.stringify(value) : String(value);
    item.append(key, text);
    details.append(item);
  };
  const strict = report.strict_fd_td_passed;
  const strictTime = report.strict_time_report && typeof report.strict_time_report === "object" ? report.strict_time_report : {};
  const checks = report.checks && typeof report.checks === "object" ? report.checks : {};
  const checksList = Array.isArray(checks) ? checks : Object.entries(checks).map(([name, value]) => ({ name, ...(value && typeof value === "object" ? value : { value }) }));
  const checkName = (check) => check?.check || check?.name || check?.key || check?.id || "";
  const namedCheck = (...names) => checksList.find((check) => names.includes(checkName(check)));
  const rmsCheck = namedCheck("native_ofdm_fd_td_web_approximation")
    || checksList.find((check) => /fd.?td.*rms|rms.*fd.?td/i.test(checkName(check)));
  const rmsRaw = report.web_rms ?? report.approximation_rms ?? report.native_fd_td_relative_rms ?? report.fd_td_relative_rms ?? report.rms
    ?? strictTime.fd_td_relative_rms_error ?? strictTime.fd_td_relative_rms
    ?? strictTime.native_fd_td_relative_rms ?? strictTime.fd_td_relative_rms ?? strictTime.relative_rms
    ?? rmsCheck?.value ?? rmsCheck?.actual ?? rmsCheck?.observed;
  const rms = finiteMetric(rmsRaw);
  const rmsThreshold = finiteMetric(
    report.web_rms_threshold,
    report.approximation_rms_threshold,
    report.rms_threshold,
    rmsCheck?.threshold,
    rmsCheck?.limit,
  ) ?? 0.001;
  const cp = namedCheck("cyclic_prefix_sufficient");
  const tapWindow = namedCheck("tap_window_outside_energy");
  const cpOutside = namedCheck("cyclic_prefix_outside_effective_energy");
  const cfr = namedCheck("direct_cfr_vs_finite_tap_cfr")
    || checksList.find((check) => /cfr/i.test(checkName(check)));
  const convergence = report.convergence;
  const convergenceChecks = Array.isArray(convergence?.checks) ? convergence.checks : [];
  const pathChecks = convergenceChecks.filter((check) => check.check === "path_coefficient_convergence");
  const maxDelayError = pathChecks.length
    ? Math.max(...pathChecks.map((check) => finiteMetric(check.delay_relative_error) ?? 0)) : null;
  const maxCoefficientError = pathChecks.length
    ? Math.max(...pathChecks.map((check) => finiteMetric(check.coefficient_relative_error) ?? 0)) : null;
  const summedCfr = convergenceChecks.find((check) => check.check === "summed_cfr_convergence");
  const convergenceSummary = convergence && typeof convergence === "object"
    ? [
      convergence.passed === true ? "通过" : convergence.passed === false ? "失败" : "诊断",
      Array.isArray(convergence.samples_per_src) ? `${convergence.samples_per_src.join("→")} samples` : "",
      pathChecks.length ? `${pathChecks.length} 条路径` : "",
      maxDelayError === null ? "" : `最大 delay 相对误差 ${formatMetric(maxDelayError, 8)}`,
      maxCoefficientError === null ? "" : `最大系数相对误差 ${formatMetric(maxCoefficientError, 8)}`,
      finiteMetric(summedCfr?.relative_error) === null
        ? "" : `汇总 CFR 相对误差 ${formatMetric(summedCfr.relative_error, 8)}`,
      convergence.error || "",
    ].filter(Boolean).join(" · ")
    : convergence;
  const reflection = report.reflection_oracle;
  const reflectionSummary = reflection && typeof reflection === "object"
    ? reflection.applicable === false || reflection.status === "not_applicable"
      ? `不适用：${reflection.reason || "无解析 oracle"}`
      : `${reflection.passed === true ? "通过" : reflection.passed === false ? "失败" : "诊断"} · ${Array.isArray(reflection.checks) ? `${reflection.checks.filter((check) => check.passed === true).length}/${reflection.checks.length} 条路径检查` : reflection.error || ""}`
    : reflection;
  const reportError = state.rtValidationSelectedJobId === job.id ? state.rtValidationError : "";
  const reportLoading = Boolean(runDownloadUrl(job, "output_validation", "validation.json"))
    && state.rtValidationReportRequestedJobId === job.id
    && state.rtValidationReportJobId !== job.id;
  add("完整检查明细", reportError ? `读取失败：${reportError}` : reportLoading ? "正在加载准入报告…" : undefined);
  add("准入策略", report.policy || report.validation_policy);
  add("Web 准入", report.passed === true ? "通过" : report.passed === false ? "未通过" : undefined);
  add("准入警告", Array.isArray(report.warnings) ? report.warnings.join("；") : report.warnings);
  add("Web 近似 RMS", rms === null ? undefined : `${formatMetric(rms, 8)}（阈值 ${formatMetric(rmsThreshold, 6)}）`);
  add("严格 FD/TD", strict === true ? "通过" : strict === false ? "未通过（保留严格报告原义）" : "未提供");
  add("CP 充分性", cp ? `${cp.passed === true ? "通过" : cp.passed === false ? "失败" : cp.reason || "诊断"}${cp.actual === undefined ? "" : ` · actual=${JSON.stringify(cp.actual)}`}` : report.cp_sufficient);
  add("tap-window 截断能量", tapWindow ? JSON.stringify(tapWindow) : undefined);
  add("CP 外有效能量", cpOutside ? JSON.stringify(cpOutside) : undefined);
  add("直接 CFR 截断误差", cfr ? JSON.stringify(cfr) : report.direct_cfr_relative_error);
  add("追踪收敛", convergenceSummary);
  add("反射 oracle", reflectionSummary);
  add("追踪变体", job.trace_variant || job.metadata?.trace_variant || report.trace_variant || report.trace_variant_used);
  add("PHY 设备", job.device || job.phy_device || job.simulation_device || job.metadata?.device || job.metadata?.phy_device);
  add("scene bundle SHA-256", report.scene_bundle_sha256 || job.scene_bundle_sha256 || job.metadata?.scene_bundle_sha256);
  add("snapshot SHA-256", report.snapshot_hash || report.snapshot_sha256 || job.snapshot_hash || job.metadata?.snapshot_hash);
  add("scene 来源", report.scene_source || job.scene_source || job.metadata?.scene_source);
  checksList.forEach((check) => {
    const label = checkName(check);
    if (!label) return;
    const outcome = check.passed === true ? "通过"
      : check.passed === false ? "失败"
      : check.reason || check.message || check.value || check.status || "诊断";
    const actual = check.actual === undefined || check.actual === null
      ? "" : ` · actual=${JSON.stringify(check.actual)}`;
    add(`检查：${label}`, `${outcome}${actual}`);
  });
  const warning = $("rt-strict-warning");
  warning.classList.toggle("hidden", strict !== false);
  warning.textContent = strict === false
    ? `严格 FD/TD 检查未通过；本结果仍是固定 RT 快照的条件 BLER，仅按网页有界频域门槛准入。${rms === null ? "" : ` RMS ${formatMetric(rms, 8)}，网页阈值 ${formatMetric(rmsThreshold, 6)}。`}`
    : "";
  const downloads = [
    ["download-validation", runDownloadUrl(job, "output_validation", "validation.json")],
    ["download-snapshot", runDownloadUrl(job, "output_snapshot_npz", "channel_snapshot.npz")],
    ["download-snapshot-json", runDownloadUrl(job, "output_snapshot_json", "channel_snapshot.json")],
    ["download-repro", runDownloadUrl(job, "output_reproducibility", "reproducibility.zip")],
  ];
  downloads.forEach(([id, url]) => {
    const link = $(id);
    link.classList.toggle("hidden", !url);
    if (url) link.href = url;
    else link.removeAttribute("href");
  });
}

function renderJob(job) {
  if (!job) return;
  state.currentJob = job;
  const backend = job.channel_backend || job.backend || (job.config_names?.rt ? "rt" : "cdl");
  state.resultBackend = backend;
  const isRt = backend === "rt";
  const active = job.status === "queued" || job.status === "running";
  const stages = { tracing: "追踪中", validating: "准入检查中", simulating: "仿真中", done: "已完成" };
  const stage = job.stage || job.stage_name;
  $("status-pill").textContent = active && stage ? stages[stage] || statusLabels[job.status] || job.status : statusLabels[job.status] || job.status || "未知状态";
  $("status-pill").className = `status-pill ${job.status || "idle"}`;
  $("cancel-button").classList.toggle("hidden", !active);
  $("job-id").textContent = job.id || "—";
  $("job-created").textContent = formatTime(job.created_at);
  $("job-duration").textContent = formatDuration(job);
  const allPoints = Array.isArray(job.points) ? job.points : [];
  const skippedCount = isRt
    ? allPoints.filter((point) => point && point.user === "all" && point.status === "skipped").length
    : allPoints.filter((point) => point && point.skipped).length;
  const pointCount = isRt
    ? allPoints.filter((point) => point && point.user === "all").length
    : allPoints.length - skippedCount;
  const totalPoints = Number(job.total_points) || 0;
  const donePoints = isRt && Number.isFinite(Number(job.completed_points))
    ? Number(job.completed_points) : pointCount + skippedCount;
  $("job-points").textContent = totalPoints
    ? `${donePoints} / ${totalPoints}${skippedCount ? `（跳过 ${skippedCount}）` : ""}`
    : String(donePoints);
  $("progress-bar").style.width = totalPoints
    ? `${Math.min(100, donePoints / totalPoints * 100)}%`
    : job.status === "completed" ? "100%" : "0";
  for (const [key, url] of [["download-csv", job.output_csv], ["download-json", job.output_manifest]]) {
    const link = $(key);
    link.classList.toggle("hidden", !url);
    if (url) link.href = url;
  }
  const logs = Array.isArray(job.logs) ? job.logs : [];
  const outputNotes = [];
  if (isRt && typeof job.snapshot_cache_hit === "boolean") {
    outputNotes.push(`静态信道快照缓存：${job.snapshot_cache_hit ? "命中" : "未命中"}`);
  }
  if (isRt && job.output_archive) outputNotes.push(`结果归档：${job.output_archive}`);
  outputNotes.push(logs.length ? logs.join("\n") : active ? "任务已提交，等待日志…" : (job.error || "暂无运行日志。"));
  $("log-view").textContent = outputNotes.join("\n");
  $("log-view").scrollTop = $("log-view").scrollHeight;
  if (job.error) showAlert(job.error);
  renderRtValidationInfo(job);
  renderResults(allPoints, job);
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
    const validation = run.validation_summary || run.web_validation || run.validation || run.metadata?.web_validation || {};
    const isRt = run.channel_backend === "rt" || run.backend === "rt" || Boolean(run.config_names?.rt);
    const strict = validation.strict_fd_td_passed ?? run.strict_fd_td_passed ?? run.metadata?.strict_fd_td_passed;
    const qualityWarning = isRt && strict === false ? "严格 FD/TD 未通过；固定快照条件 BLER" : "";
    const cells = [
      ["任务", run.id || "—", "history-id"],
      ["状态", statusLabels[run.status] || run.status || "—"],
      ["创建时间", formatTime(run.created_at)],
      ["配置", summarizeConfigs(run.config_names), "history-config", qualityWarning],
    ];
    cells.forEach(([label, value, className, warning]) => {
      const cell = document.createElement("div");
      const labelEl = document.createElement("span");
      labelEl.className = "history-label";
      labelEl.textContent = label;
      const valueEl = document.createElement("span");
      valueEl.className = className || "history-value";
      valueEl.textContent = value;
      cell.append(labelEl, valueEl);
      if (warning) {
        const warningEl = document.createElement("small");
        warningEl.className = "rt-history-warning";
        warningEl.textContent = warning;
        cell.append(warningEl);
      }
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
  return [names.tx, names.channel || names.rt, names.simulation].filter(Boolean).join(" · ") || "—";
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
function cloneValue(value) {
  return value == null ? value : JSON.parse(JSON.stringify(value));
}

function tomlSection(text, sectionName) {
  const values = {};
  let current = "";
  for (const line of String(text || "").split(/\r?\n/)) {
    const header = line.match(/^\s*\[([^\]]+)\]\s*$/);
    if (header) {
      current = header[1];
      continue;
    }
    if (current !== sectionName) continue;
    const entry = line.match(/^\s*([A-Za-z0-9_]+)\s*=\s*(.*?)\s*(?:#.*)?$/);
    if (!entry) continue;
    const raw = entry[2].trim();
    try { values[entry[1]] = JSON.parse(raw); }
    catch {
      if (raw === "true" || raw === "false") values[entry[1]] = raw === "true";
      else if (/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(raw)) values[entry[1]] = Number(raw);
      else values[entry[1]] = raw.replace(/^["']|["']$/g, "");
    }
  }
  return values;
}

function rtScenarioProfile(scenario, kind) {
  return scenario?.profiles?.[kind] || scenario?.[`${kind}_name`] || scenario?.[kind] || "";
}

function rtScenarios() {
  return state.scenarios.filter(isRtScenario);
}

function configOption(select, name, text = name) {
  const option = document.createElement("option");
  option.value = name;
  option.textContent = text;
  select.append(option);
}

function populateRtProfileSelects() {
  const pairs = [
    [$("rt-profile-select"), state.configs.rt, state.rtSelected.rt],
    [$("rt-tx-profile-select"), state.configs.tx, state.rtSelected.tx],
    [$("rt-simulation-profile-select"), state.configs.simulation, state.rtSelected.simulation],
  ];
  pairs.forEach(([select, names, selected]) => {
    select.replaceChildren();
    (names || []).forEach((name) => configOption(select, name));
    select.disabled = !names?.length;
    if (selected) select.value = selected;
  });
}

function deviceNames(value, output = []) {
  if (typeof value === "string") output.push(value.toLowerCase());
  else if (Array.isArray(value)) value.forEach((item) => deviceNames(item, output));
  else if (value && typeof value === "object") {
    const name = value.name || value.device || value.id;
    const available = value.available ?? value.is_available ?? true;
    if (typeof name === "string" && available) output.push(name.toLowerCase());
    Object.entries(value).forEach(([key, child]) => {
      if (/^(cpu|cuda(?::\d+)?)$/i.test(key)) {
        const childAvailable = child && typeof child === "object" ? child.available ?? child.is_available ?? true : Boolean(child);
        if (childAvailable) output.push(key.toLowerCase());
      } else if (child && typeof child === "object") deviceNames(child, output);
    });
  }
  return output;
}

function availableRtDevices() {
  const devices = [...new Set(deviceNames(state.rtOptions?.devices).filter((name) => /^(cpu|cuda(?::\d+)?)$/.test(name)))];
  if (!devices.includes("cpu")) devices.unshift("cpu");
  return devices.sort((left, right) => left === "cpu" ? -1 : right === "cpu" ? 1 : left.localeCompare(right));
}
function deviceIsAvailable(name) {
  const lower = String(name).toLowerCase();
  const devices = availableRtDevices();
  return lower === "cpu" || devices.includes(lower) || (lower === "cuda" && devices.some((device) => device.startsWith("cuda:")));
}

function updateRtDeviceSelector() {
  const select = $("rt-phy-device");
  if (!select) return;
  const selected = tomlSection(state.rtSimulationText, "bler").device || "cpu";
  const available = availableRtDevices();
  select.replaceChildren();
  const choices = new Set(available);
  if (!choices.has(String(selected).toLowerCase())) choices.add(String(selected).toLowerCase());
  [...choices].forEach((name) => {
    configOption(select, name);
    const option = [...select.options].at(-1);
    option.disabled = !deviceIsAvailable(name);
  });
  select.value = String(selected).toLowerCase();
}

function updateBlerDeviceToml(text, device) {
  const lines = String(text || "").split(/\r?\n/);
  let sectionStart = lines.findIndex((line) => /^\s*\[bler\]\s*(?:#.*)?$/.test(line));
  if (sectionStart < 0) {
    const prefix = String(text || "").trimEnd();
    return `${prefix}${prefix ? "\n\n" : ""}[bler]\ndevice = ${JSON.stringify(device)}\n`;
  }
  let sectionEnd = lines.findIndex((line, index) => index > sectionStart && /^\s*\[/.test(line));
  if (sectionEnd < 0) sectionEnd = lines.length;
  for (let index = sectionStart + 1; index < sectionEnd; index++) {
    if (/^\s*device\s*=/.test(lines[index])) {
      lines[index] = `${lines[index].match(/^\s*/)[0]}device = ${JSON.stringify(device)}`;
      return lines.join("\n");
    }
  }
  lines.splice(sectionEnd, 0, `device = ${JSON.stringify(device)}`);
  return lines.join("\n");
}

async function setRtPhyDevice(device) {
  const oldDevice = tomlSection(state.rtSimulationText, "bler").device || "cpu";
  if (!deviceIsAvailable(device)) {
    $("rt-phy-device").value = oldDevice;
    showAlert(`设备 ${device} 当前不可用。`);
    return;
  }
  state.rtSimulationText = updateBlerDeviceToml(state.rtSimulationText, device);
  state.rtDirty = true;
  state.rtScenarioId = "";
  populateRtPresetSelect();
  updateRtSummary();
  updateRtBudgetAvailability();
  state.validation = null;
  renderCompatibility(null);
  try { await validateCurrentRun(); }
  catch (error) {
    state.validation = null;
    renderCompatibility({ valid: false, errors: [{ kind: "validation", message: error.message }] });
  }
}

function cudaAvailable() {
  return availableRtDevices().some((name) => name.startsWith("cuda"));
}

function updateRtBudgetAvailability() {
  const select = $("rt-budget");
  if (!select) return;
  const longName = "bler_rt_beam.toml";
  const hasLongProfile = state.configs.simulation.includes(longName);
  const available = cudaAvailable() && hasLongProfile;
  const longOption = [...select.options].find((option) => option.value === "long");
  if (longOption) {
    longOption.disabled = !available;
    longOption.title = available ? "使用较长统计配置和 CUDA 设备" : "CUDA 不可用或缺少较长统计配置";
  }
  let customOption = [...select.options].find((option) => option.value === "custom");
  if (state.rtBudget === "custom" && !customOption) {
    configOption(select, "custom", "高级自定义预算");
    customOption = [...select.options].find((option) => option.value === "custom");
  } else if (state.rtBudget !== "custom" && customOption) customOption.remove();
  if (customOption) customOption.disabled = true;
  if (state.rtBudget === "long") select.value = "long";
  else if (state.rtBudget === "custom") select.value = "custom";
  else {
    state.rtBudget = "quick";
    select.value = "quick";
  }
  const deviceMessage = state.rtOptionsError
    ? `设备能力查询失败：${state.rtOptionsError}。${available ? "" : "仅保留 CPU 快速诊断；较长统计已禁用。"}`
    : available
      ? `可用设备：${availableRtDevices().join("、")}。仿真实际使用所选 PHY 设备；不会自动切换设备。`
      : "CUDA 不可用或较长统计配置缺失；较长统计选项已禁用。可在高级配置中显式选择 CPU，不会自动迁移设备或缩减预算。";
  $("rt-device-status").textContent = deviceMessage;
  $("rt-budget-guidance").textContent = available
    ? "固定几何、静态快照的条件 BLER；不代表严格时域等价或真实城区 benchmark。"
    : "快速诊断可在 CPU 上运行。较长统计需要 CUDA；条件 BLER 不代表严格时域等价或真实城区 benchmark。";
  updateRtDeviceSelector();
}

async function initRtCatalog() {
  state.rtScenarios = rtScenarios();
  try {
    state.rtOptions = await api("/api/rt/options");
    state.rtOptionsError = "";
  } catch (error) {
    state.rtOptions = { devices: ["cpu"] };
    state.rtOptionsError = parseRtError(error);
  }
  populateRtSceneKindSelect();
  populateRtPresetSelect();
  updateRtBudgetAvailability();
}

function rtBuiltinSceneIds() {
  const presets = state.rtOptions?.scene_presets;
  return new Set(
    Array.isArray(presets)
      ? presets.map((preset) => preset?.id).filter((id) => typeof id === "string")
      : ["empty", "ground", "ground_wall"],
  );
}

function populateRtSceneKindSelect() {
  const select = $("rt-scene-kind");
  if (!select) return;
  const presets = state.rtOptions?.scene_presets;
  if (!Array.isArray(presets) || !presets.length) return;
  const selected = select.value;
  select.replaceChildren();
  presets.forEach((preset) => {
    configOption(select, preset.id, preset.label || preset.id);
    select.options[select.options.length - 1].title = preset.description || "";
  });
  configOption(select, "custom", "导入 ZIP");
  const validIds = rtBuiltinSceneIds();
  select.value = selected === "custom" || validIds.has(selected) ? selected : "empty";
}

function populateRtPresetSelect() {
  const select = $("rt-preset-select");
  if (!select) return;
  select.replaceChildren();
  configOption(select, "custom", "自定义 RT 配置");
  state.rtScenarios.forEach((scenario) => configOption(select, scenario.id, scenario.label || scenario.id));
  select.value = state.rtScenarioId || "custom";
}

function updateRtRunButton() {
  $("run-button").textContent = "";
  const play = document.createElement("span");
  play.className = "play";
  play.setAttribute("aria-hidden", "true");
  $("run-button").append(play, document.createTextNode(state.backend === "rt" ? "开始 RT BLER 仿真" : "启动 BLER 仿真"));
  $("run-hint").textContent = state.backend === "rt"
    ? "一次启动场景追踪、准入检查与 BLER 仿真"
    : "先选择场景并通过兼容性检查";
}

async function switchBackend(backend) {
  state.backend = backend === "rt" ? "rt" : "cdl";
  const isRt = state.backend === "rt";
  $("cdl-config").classList.toggle("hidden", isRt);
  $("rt-config").classList.toggle("hidden", !isRt);
  $("backend-cdl-button").classList.toggle("active", !isRt);
  $("backend-cdl-button").setAttribute("aria-pressed", String(!isRt));
  $("backend-rt-button").classList.toggle("active", isRt);
  $("backend-rt-button").setAttribute("aria-pressed", String(isRt));
  $("backend-note").textContent = isRt
    ? "Sionna RT 使用固定几何快照与网页准入门槛；运行会自动追踪、检查并仿真。"
    : "CDL 工作流保持原有配置与接收分析行为。";
  updateRtRunButton();
  state.validation = null;
  renderCompatibility(null);
  if (isRt) {
    if (!state.rtOptions) await initRtCatalog();
    if (!state.rtInitialized) {
      const initial = state.rtScenarios.find((scenario) => scenario.id === "rt-los-quick") || state.rtScenarios[0];
      if (!initial) throw new Error("RT 场景列表中没有可用预设。");
      await loadRtScenario(initial.id, false);
    } else {
      renderRtSettings();
      await validateCurrentRun();
    }
  } else {
    await validateCurrentRun();
    if ($("advanced-config").open) await refreshProfileCompatibility();
  }
}

function rtGeometryDefaults(scene) {
  const geometry = {
    ground_bounds_m: [-500, 500, -500, 500],
    ground_height_m: 0,
    ground_material: "concrete",
    ground_thickness_m: 0.1,
  };
  if (scene === "ground_wall") {
    Object.assign(geometry, {
      wall_start_xy_m: [-100, 150],
      wall_end_xy_m: [500, 150],
      wall_base_height_m: 0,
      wall_height_m: 50,
      wall_material: "brick",
      wall_thickness_m: 0.1,
    });
  }
  return geometry;
}

async function readRtProfile(kind, name) {
  if (!name) throw new Error(`未选择 ${kind} 配置文件。`);
  return api(`/api/configs/${encodeURIComponent(kind)}/${encodeURIComponent(name)}`);
}

async function loadRtScenario(scenarioId, confirmDirty = true) {
  const scenario = state.rtScenarios.find((item) => item.id === scenarioId);
  if (!scenario) return;
  if (confirmDirty && state.rtDirty && !window.confirm("切换 RT 预设会丢弃当前自定义修改。继续吗？")) {
    $("rt-preset-select").value = state.rtScenarioId || "custom";
    return;
  }
  clearTimeout(state.rtConfigTimer);
  state.rtConfigTimer = null;
  const loadSerial = ++state.rtConfigSerial;
  state.rtConfigPending = true;
  state.rtConfigPromise = null;
  state.validation = null;
  renderCompatibility(null);
  const names = {
    tx: rtScenarioProfile(scenario, "tx"),
    rt: rtScenarioProfile(scenario, "rt"),
    simulation: rtScenarioProfile(scenario, "simulation"),
  };
  let tx, rt, simulation, parsed;
  try {
    [tx, rt, simulation] = await Promise.all([
      readRtProfile("tx", names.tx),
      readRtProfile("rt", names.rt),
      readRtProfile("simulation", names.simulation),
    ]);
    parsed = await api("/api/rt/config", {
      method: "POST",
      body: JSON.stringify({ rt_text: rt.text || "" }),
    });
  } catch (error) {
    if (loadSerial === state.rtConfigSerial) {
      state.rtConfigPending = false;
      state.rtConfigPromise = null;
      $("rt-apply-toml").disabled = false;
      populateRtPresetSelect();
      if (state.backend === "rt") {
        try { await validateCurrentRun(); }
        catch (validationError) {
          state.validation = null;
          renderCompatibility({ valid: false, errors: [{ kind: "validation", message: validationError.message }] });
        }
      }
    }
    throw error;
  }
  if (loadSerial !== state.rtConfigSerial) return;
  state.rtConfigPending = false;
  state.rtConfigPromise = null;
  $("rt-apply-toml").disabled = false;
  state.rtSelected = names;
  state.rtTxText = tx.text || "";
  state.rtText = rt.text || parsed.rt_text || "";
  state.rtSimulationText = simulation.text || "";
  state.rtSettings = parsed.rt_settings;
  state.rtDraftSettings = cloneValue(parsed.rt_settings);
  state.rtPresetId = scenario.id;
  state.rtScenarioId = scenario.id;
  state.rtSceneId = null;
  state.rtSceneFilename = "";
  state.rtSceneBundleHash = "";
  state.rtMeshSummary = null;
  state.rtDirty = false;
  state.rtExpertDirty = false;
  state.rtConfigError = "";
  state.rtInitialized = true;
  state.rtWaveform = names.tx === "pusch_rt_4ue_cp.toml" || /cp[_-]ofdm/i.test(tomlSection(state.rtTxText, "pusch").waveform || "") ? "cp" : "dft";
  state.rtBudget = names.simulation === "bler_rt_beam.toml" ? "long"
    : names.simulation === "bler_rt_beam_web_quick.toml" ? "quick" : "custom";
  $("rt-waveform").value = state.rtWaveform;
  $("rt-budget").value = state.rtBudget;
  $("rt-expert-editor").value = state.rtText;
  $("rt-expert-status").textContent = "预设 TOML 已载入；当前预设身份保持有效。";
  populateRtProfileSelects();
  populateRtPresetSelect();
  renderRtSettings();
  updateRtBudgetAvailability();
  await validateCurrentRun();
}

function rtSource(settings = state.rtDraftSettings || state.rtSettings) {
  if (!settings) return "builtin";
  if (settings.rt?.scene === "custom") return "imported";
  if (settings.geometry) return "parameterized";
  return "builtin";
}

function rtValueAtPath(settings, path) {
  return path.split(".").reduce((value, key) => value == null ? undefined : value[key], settings);
}

function writeRtValueAtPath(settings, path, value) {
  const parts = path.split(".");
  let target = settings;
  for (let index = 0; index < parts.length - 1; index++) {
    const key = parts[index];
    const nextIsIndex = /^\d+$/.test(parts[index + 1]);
    if (!target[key] || typeof target[key] !== "object") target[key] = nextIsIndex ? [] : {};
    target = target[key];
  }
  target[parts[parts.length - 1]] = value;
}

function renderRtSettings() {
  const settings = state.rtDraftSettings || state.rtSettings;
  if (!settings) return;
  $("rt-fields").querySelectorAll("[data-rt-path]").forEach((input) => {
    const value = rtValueAtPath(settings, input.dataset.rtPath);
    if (value !== undefined && value !== null) input.value = String(value);
    else if (input.tagName !== "SELECT") input.value = "";
  });
  const source = rtSource(settings);
  $("rt-scene-source").value = source;
  $("rt-scene-kind").value = settings.rt?.scene || "empty";
  const imported = source === "imported";
  $("rt-scene-kind").disabled = imported;
  [...$("rt-scene-kind").options].forEach((option) => {
    option.disabled = source === "imported"
      || (source === "parameterized" && !["ground", "ground_wall"].includes(option.value))
      || (source !== "imported" && option.value === "custom");
  });
  const hasParameterizedWall = source === "parameterized" && settings.rt?.scene === "ground_wall";
  $("rt-geometry-fields").classList.toggle("hidden", source !== "parameterized");
  $("rt-wall-heading").classList.toggle("hidden", !hasParameterizedWall);
  $("rt-wall-fields").classList.toggle("hidden", !hasParameterizedWall);
  $("rt-geometry-fields").querySelectorAll('[data-rt-path^="geometry.wall_"]').forEach((input) => {
    input.disabled = !hasParameterizedWall;
  });
  $("rt-expert-editor").value = state.rtText;
  $("rt-tx-editor").value = state.rtTxText;
  $("rt-simulation-editor").value = state.rtSimulationText;
  $("rt-config-error").classList.toggle("hidden", !state.rtConfigError);
  $("rt-config-error").textContent = state.rtConfigError;
  $("rt-user-filter").value = state.rtResultUser;
  updateRtSceneStatus();
  updateRtBudgetAvailability();
  updateRtSummary();
  drawRtPreview();
}

function updateRtSceneStatus() {
  const settings = state.rtDraftSettings || state.rtSettings;
  if (state.rtSceneId) {
    $("rt-scene-status").textContent = `已导入 ${state.rtSceneFilename || "场景包"} · scene_id ${state.rtSceneId} · bundle ${state.rtSceneBundleHash || "—"}`;
  } else if (settings?.geometry) {
    $("rt-scene-status").textContent = `参数化场景：${settings.rt?.scene === "ground_wall" ? "地面＋墙面" : "地面"}；由服务端为任务生成 canonical XML＋PLY。`;
  } else if (settings?.rt?.scene === "custom") {
    $("rt-scene-status").textContent = "导入场景包：尚未上传 ZIP；请先选择并上传合法场景包。";
  } else {
    const scene = settings?.rt?.scene;
    const preset = state.rtOptions?.scene_presets?.find((item) => item.id === scene);
    $("rt-scene-status").textContent = preset
      ? `内置场景：${preset.label}。${preset.description}`
      : `内置场景：${scene || "LoS"}`;
  }
  const entries = state.rtMeshSummary && typeof state.rtMeshSummary === "object" ? Object.entries(state.rtMeshSummary) : [];
  $("rt-mesh-summary").textContent = entries.length
    ? entries.map(([name, mesh]) => {
      const bounds = mesh?.bounds_xy_m;
      const min = bounds?.min, max = bounds?.max;
      const box = Array.isArray(min) && Array.isArray(max) ? ` · XY [${min.join(", ")}]–[${max.join(", ")}] m` : "";
      return `${name}: ${mesh.vertex_count ?? "?"} 顶点 / ${mesh.face_count ?? "?"} 面${box}`;
    }).join("；")
    : "";
}

function tomlArrayOr(value, fallback = []) {
  return Array.isArray(value) ? value : fallback;
}

function updateRtSummary() {
  const settings = state.rtDraftSettings || state.rtSettings;
  if (!settings) return;
  const simulation = tomlSection(state.rtSimulationText, "bler");
  const snrs = tomlArrayOr(simulation.snr_db, []);
  const estimators = tomlArrayOr(simulation.channel_estimators, simulation.channel_estimator ? [simulation.channel_estimator] : []);
  const detectors = tomlArrayOr(simulation.detectors, simulation.detector ? [simulation.detector] : []);
  const users = Array.isArray(settings.users) ? settings.users.length : 0;
  const rows = Number(settings.receiver?.num_rows) || 0;
  const cols = Number(settings.receiver?.num_cols) || 0;
  const frequency = Number(settings.rt?.carrier_frequency_hz);
  const ratio = Number(settings.noise?.post_combiner_ratio);
  const frameCount = simulation.max_frames_per_snr ?? "—";
  const device = simulation.device || "未指定";
  const summary = $("rt-summary");
  summary.replaceChildren();
  const cards = [
    ["链路", `${users} UE · 单层 / 单端口`],
    ["阵列", `${rows} × ${cols} · ${rows * cols} 天线`],
    ["载频", Number.isFinite(frequency) ? `${formatMetric(frequency / 1e9, 2)} GHz` : "—"],
    ["后级噪声比", Number.isFinite(ratio) ? formatMetric(ratio, 6) : "—"],
    ["SNR / 每点帧数", `${snrs.length ? snrs.map((snr) => `${snr} dB`).join("、") : "—"} / ${frameCount}`],
    ["算法 / PHY 设备", `${estimators.length} CSI × ${detectors.length} 检测器 · ${device}`],
  ];
  cards.forEach(([label, value]) => {
    const card = document.createElement("div");
    const caption = document.createElement("span");
    caption.textContent = label;
    const strong = document.createElement("strong");
    strong.textContent = value;
    card.append(caption, strong);
    summary.append(card);
  });
  $("rt-device-status").dataset.configuredDevice = device;
  drawRtPreview();
}

function rtMeshBounds() {
  const summary = state.rtMeshSummary;
  if (!summary || typeof summary !== "object") return [];
  return Object.entries(summary).flatMap(([name, mesh]) => {
    const bounds = mesh?.bounds_xy_m;
    if (!Array.isArray(bounds?.min) || !Array.isArray(bounds?.max) || bounds.min.length < 2 || bounds.max.length < 2) return [];
    const values = [...bounds.min.slice(0, 2), ...bounds.max.slice(0, 2)].map(Number);
    if (!values.every(Number.isFinite)) return [];
    return [{ name, x0: values[0], y0: values[1], x1: values[2], y1: values[3] }];
  });
}

function drawRtPreview() {
  const canvas = $("rt-scene-preview");
  const settings = state.rtDraftSettings || state.rtSettings;
  if (!canvas || !settings) return;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const width = rect.width, height = rect.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfcf8";
  ctx.fillRect(0, 0, width, height);
  const bs = settings.receiver?.position_m || [0, 0, 0];
  const users = Array.isArray(settings.users) ? settings.users : [];
  const geometry = settings.geometry;
  const ground = geometry?.ground_bounds_m;
  const meshes = rtMeshBounds();
  const xs = [Number(bs[0]) || 0, ...users.map((user) => Number(user.position_m?.[0]) || 0)];
  const ys = [Number(bs[1]) || 0, ...users.map((user) => Number(user.position_m?.[1]) || 0)];
  if (Array.isArray(ground) && ground.length === 4) { xs.push(Number(ground[0]), Number(ground[1])); ys.push(Number(ground[2]), Number(ground[3])); }
  meshes.forEach((box) => { xs.push(box.x0, box.x1); ys.push(box.y0, box.y1); });
  const start = geometry?.wall_start_xy_m, end = geometry?.wall_end_xy_m;
  if (Array.isArray(start)) { xs.push(Number(start[0])); ys.push(Number(start[1])); }
  if (Array.isArray(end)) { xs.push(Number(end[0])); ys.push(Number(end[1])); }
  let xMin = Math.min(...xs.filter(Number.isFinite)), xMax = Math.max(...xs.filter(Number.isFinite));
  let yMin = Math.min(...ys.filter(Number.isFinite)), yMax = Math.max(...ys.filter(Number.isFinite));
  if (xMin === xMax) { xMin -= 10; xMax += 10; }
  if (yMin === yMax) { yMin -= 10; yMax += 10; }
  const pad = { left: 40, right: 18, top: 16, bottom: 32 };
  const scale = Math.min((width - pad.left - pad.right) / (xMax - xMin), (height - pad.top - pad.bottom) / (yMax - yMin));
  const plotWidth = (xMax - xMin) * scale, plotHeight = (yMax - yMin) * scale;
  const offsetX = ((width - pad.left - pad.right) - plotWidth) / 2;
  const offsetY = ((height - pad.top - pad.bottom) - plotHeight) / 2;
  const plotLeft = pad.left + offsetX, plotRight = width - pad.right - offsetX;
  const plotTop = pad.top + offsetY, plotBottom = height - pad.bottom - offsetY;
  const px = (x) => plotLeft + (x - xMin) * scale;
  const py = (y) => plotBottom - (y - yMin) * scale;
  ctx.strokeStyle = "#e7ece7";
  ctx.lineWidth = 1;
  for (let index = 0; index <= 5; index++) {
    const gx = plotLeft + plotWidth * index / 5;
    const gy = plotTop + plotHeight * index / 5;
    ctx.beginPath(); ctx.moveTo(gx, plotTop); ctx.lineTo(gx, plotBottom); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(plotLeft, gy); ctx.lineTo(plotRight, gy); ctx.stroke();
  }
  ctx.font = "9px ui-monospace, monospace";
  ctx.fillStyle = "#77837d";
  ctx.textAlign = "center";
  ctx.fillText("x (m)", plotLeft + plotWidth / 2, height - 8);
  ctx.save(); ctx.translate(11, plotTop + plotHeight / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("y (m)", 0, 0); ctx.restore();
  if (Array.isArray(ground) && ground.length === 4) {
    ctx.fillStyle = "rgba(133, 173, 82, .12)";
    ctx.strokeStyle = "#8ba75f";
    ctx.fillRect(px(Number(ground[0])), py(Number(ground[3])), (Number(ground[1]) - Number(ground[0])) * scale, (Number(ground[3]) - Number(ground[2])) * scale);
    ctx.strokeRect(px(Number(ground[0])), py(Number(ground[3])), (Number(ground[1]) - Number(ground[0])) * scale, (Number(ground[3]) - Number(ground[2])) * scale);
  }
  meshes.forEach((box) => {
    ctx.fillStyle = "rgba(86, 122, 164, .12)";
    ctx.strokeStyle = "#718eab";
    ctx.fillRect(px(box.x0), py(box.y1), (box.x1 - box.x0) * scale, (box.y1 - box.y0) * scale);
    ctx.strokeRect(px(box.x0), py(box.y1), (box.x1 - box.x0) * scale, (box.y1 - box.y0) * scale);
    ctx.fillStyle = "#526a82"; ctx.textAlign = "left"; ctx.fillText(box.name.split("/").pop(), px(box.x0) + 3, py(box.y1) + 11);
  });
  if (Array.isArray(start) && Array.isArray(end)) {
    ctx.strokeStyle = "#a74c38"; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(px(Number(start[0])), py(Number(start[1]))); ctx.lineTo(px(Number(end[0])), py(Number(end[1]))); ctx.stroke();
  }
  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 1.2;
  users.forEach((user, index) => {
    const position = user.position_m || [0, 0, 0];
    ctx.strokeStyle = seriesColors[index % seriesColors.length];
    ctx.beginPath(); ctx.moveTo(px(Number(bs[0]) || 0), py(Number(bs[1]) || 0)); ctx.lineTo(px(Number(position[0]) || 0), py(Number(position[1]) || 0)); ctx.stroke();
  });
  ctx.setLineDash([]);
  users.forEach((user, index) => {
    const position = user.position_m || [0, 0, 0];
    ctx.fillStyle = seriesColors[index % seriesColors.length];
    ctx.beginPath(); ctx.arc(px(Number(position[0]) || 0), py(Number(position[1]) || 0), 4, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#24312a"; ctx.textAlign = "left"; ctx.fillText(`ue${index}`, px(Number(position[0]) || 0) + 6, py(Number(position[1]) || 0) - 5);
  });
  ctx.fillStyle = "#19251f"; ctx.fillRect(px(Number(bs[0]) || 0) - 5, py(Number(bs[1]) || 0) - 5, 10, 10);
  ctx.fillStyle = "#19251f"; ctx.textAlign = "left"; ctx.fillText("BS", px(Number(bs[0]) || 0) + 7, py(Number(bs[1]) || 0) + 12);
}


function parseRtError(error) {
  if (error && typeof error.message === "string") return error.message;
  return String(error || "RT 配置无效");
}


async function sendRtSettings(serial, settings) {
  try {
    const result = await api("/api/rt/config", {
      method: "POST",
      body: JSON.stringify({ rt_settings: settings }),
    });
    if (serial !== state.rtConfigSerial) return false;
    state.rtSettings = result.rt_settings;
    state.rtDraftSettings = cloneValue(result.rt_settings);
    state.rtText = result.rt_text || "";
    state.rtConfigPending = false;
    state.rtConfigError = "";
    state.rtScenarioId = "";
    state.rtDirty = true;
    state.rtExpertDirty = false;
    $("rt-expert-editor").value = state.rtText;
    $("rt-expert-status").textContent = "表单配置已由服务端解析；这是当前有效 RT TOML。";
    $("rt-config-error").classList.add("hidden");
    populateRtPresetSelect();
    renderRtSettings();
    if (state.backend === "rt") await validateCurrentRun();
    return true;
  } catch (error) {
    if (serial !== state.rtConfigSerial) return false;
    state.rtConfigPending = false;
    state.rtConfigError = parseRtError(error);
    $("rt-config-error").textContent = state.rtConfigError;
    $("rt-config-error").classList.remove("hidden");
    state.validation = null;
    renderCompatibility({ valid: false, errors: [{ kind: "rt", message: state.rtConfigError }] });
    return false;
  } finally {
    if (serial === state.rtConfigSerial) {
      updateRtSummary();
      if (state.validation) renderCompatibility(state.validation);
      $("rt-apply-toml").disabled = false;
    }
  }
}

function scheduleRtSettingsRequest(settings) {
  clearTimeout(state.rtConfigTimer);
  const serial = ++state.rtConfigSerial;
  state.rtDraftSettings = cloneValue(settings);
  state.rtConfigPending = true;
  state.rtConfigError = "";
  state.rtScenarioId = "";
  state.rtDirty = true;
  state.validation = null;
  populateRtPresetSelect();
  renderCompatibility(null);
  $("rt-config-error").classList.add("hidden");
  state.rtConfigPromise = new Promise((resolve) => {
    state.rtConfigTimer = setTimeout(() => {
      state.rtConfigTimer = null;
      const request = sendRtSettings(serial, state.rtDraftSettings);
      state.rtConfigPromise = request;
      request.then(resolve);
    }, 300);
  });
}

async function flushRtSettings() {
  if (state.rtExpertDirty) throw new Error("RT TOML 有未应用的修改；请先点“应用 TOML”或恢复表单配置。");
  if (state.rtConfigTimer) {
    clearTimeout(state.rtConfigTimer);
    state.rtConfigTimer = null;
    const serial = state.rtConfigSerial;
    state.rtConfigPromise = sendRtSettings(serial, state.rtDraftSettings);
  }
  if (state.rtConfigPending && state.rtConfigPromise) await state.rtConfigPromise;
  if (state.rtConfigPending) throw new Error("RT 配置解析尚未完成；请等待解析完成后再运行。");
  if (state.rtConfigError) throw new Error(state.rtConfigError);
  return true;
}

function onRtFieldInput(event) {
  const input = event.target.closest("[data-rt-path]");
  if (!input) return;
  if (state.rtExpertDirty) {
    showAlert("请先应用 RT TOML，或放弃专家编辑后再修改表单。");
    renderRtSettings();
    return;
  }
  const settings = cloneValue(state.rtDraftSettings || state.rtSettings);
  if (!settings) return;
  let value = input.value;
  if (input.type === "number") value = value.trim() === "" ? "" : Number(value);
  writeRtValueAtPath(settings, input.dataset.rtPath, value);
  state.rtDraftSettings = settings;
  scheduleRtSettingsRequest(settings);
  updateRtSummary();
  drawRtPreview();
}

function onRtSceneSourceChange() {
  if (state.rtExpertDirty) return showAlert("请先应用 RT TOML 后再编辑场景来源。");
  const settings = cloneValue(state.rtDraftSettings || state.rtSettings);
  if (!settings) return;
  const source = $("rt-scene-source").value;
  let scene = settings.rt?.scene;
  if (source === "builtin") {
    const available = rtBuiltinSceneIds();
    if (!available.has(scene)) scene = available.has("empty") ? "empty" : available.values().next().value;
    settings.rt.scene = scene;
    delete settings.rt.scene_file;
    delete settings.geometry;
  } else if (source === "parameterized") {
    if (!["ground", "ground_wall"].includes(scene)) scene = "ground";
    settings.rt.scene = scene;
    delete settings.rt.scene_file;
    settings.geometry = rtGeometryDefaults(scene);
  } else {
    settings.rt.scene = "custom";
    settings.rt.scene_file = "scene.xml";
    delete settings.geometry;
  }
  if (source !== "imported") {
    state.rtSceneId = null;
    state.rtSceneFilename = "";
    state.rtSceneBundleHash = "";
    state.rtMeshSummary = null;
  }
  state.rtDraftSettings = settings;
  scheduleRtSettingsRequest(settings);
  renderRtSettings();
}

function onRtSceneKindChange() {
  if (state.rtExpertDirty) return showAlert("请先应用 RT TOML 后再编辑场景类型。");
  const settings = cloneValue(state.rtDraftSettings || state.rtSettings);
  if (!settings) return;
  const source = $("rt-scene-source").value;
  const scene = $("rt-scene-kind").value;
  if (source === "imported") return;
  if (source === "parameterized" && !["ground", "ground_wall"].includes(scene)) return;
  settings.rt.scene = scene;
  if (source === "parameterized") settings.geometry = rtGeometryDefaults(scene);
  else {
    delete settings.geometry;
    delete settings.rt.scene_file;
  }
  state.rtSceneId = null;
  state.rtSceneFilename = "";
  state.rtSceneBundleHash = "";
  state.rtMeshSummary = null;
  state.rtDraftSettings = settings;
  scheduleRtSettingsRequest(settings);
  renderRtSettings();
}

async function applyRtToml() {
  const text = $("rt-expert-editor").value;
  clearTimeout(state.rtConfigTimer);
  state.rtConfigTimer = null;
  const serial = ++state.rtConfigSerial;
  state.rtConfigPending = true;
  state.rtConfigError = "";
  state.validation = null;
  renderCompatibility(null);
  $("rt-apply-toml").disabled = true;
  let resolveRequest;
  state.rtConfigPromise = new Promise((resolve) => { resolveRequest = resolve; });
  try {
    const result = await api("/api/rt/config", { method: "POST", body: JSON.stringify({ rt_text: text }) });
    if (serial !== state.rtConfigSerial) return;
    state.rtSettings = result.rt_settings;
    state.rtDraftSettings = cloneValue(result.rt_settings);
    state.rtText = result.rt_text || text;
    state.rtConfigPending = false;
    state.rtConfigError = "";
    state.rtExpertDirty = false;
    state.rtScenarioId = "";
    state.rtDirty = true;
    $("rt-expert-editor").value = state.rtText;
    $("rt-expert-status").textContent = "TOML 已成功解析并应用到表单。";
    populateRtPresetSelect();
    renderRtSettings();
    if (state.backend === "rt") await validateCurrentRun();
  } catch (error) {
    if (serial !== state.rtConfigSerial) return;
    state.rtConfigPending = false;
    state.rtConfigError = parseRtError(error);
    $("rt-config-error").textContent = state.rtConfigError;
    $("rt-config-error").classList.remove("hidden");
    $("rt-expert-status").textContent = "解析失败；输入已保留，未覆盖当前有效配置。";
    state.validation = null;
    renderCompatibility({ valid: false, errors: [{ kind: "rt", message: state.rtConfigError }] });
  } finally {
    resolveRequest();
    if (serial === state.rtConfigSerial) $("rt-apply-toml").disabled = false;
  }
}

async function changeRtProfile(kind, name) {
  if (!name) return;
  if (state.rtDirty && !window.confirm("切换 RT 配置文件会丢弃当前 RT 自定义修改。继续吗？")) {
    const selectId = kind === "rt" ? "rt-profile-select" : kind === "tx" ? "rt-tx-profile-select" : "rt-simulation-profile-select";
    $(selectId).value = state.rtSelected[kind];
    return;
  }
  clearTimeout(state.rtConfigTimer);
  state.rtConfigTimer = null;
  const serial = ++state.rtConfigSerial;
  const previousError = state.rtConfigError;
  state.rtConfigPending = true;
  state.rtConfigPromise = null;
  state.validation = null;
  renderCompatibility(null);
  try {
    const profile = await readRtProfile(kind, name);
    if (kind === "rt") {
      const parsed = await api("/api/rt/config", { method: "POST", body: JSON.stringify({ rt_text: profile.text || "" }) });
      if (serial !== state.rtConfigSerial) return;
      state.rtSettings = parsed.rt_settings;
      state.rtDraftSettings = cloneValue(parsed.rt_settings);
      state.rtText = profile.text || parsed.rt_text || "";
      state.rtExpertDirty = false;
      $("rt-expert-editor").value = state.rtText;
      state.rtSceneId = null;
      state.rtSceneFilename = "";
      state.rtSceneBundleHash = "";
      state.rtMeshSummary = null;
    } else if (kind === "tx") state.rtTxText = profile.text || "";
    else {
      state.rtSimulationText = profile.text || "";
      state.rtBudget = name === "bler_rt_beam.toml" ? "long"
        : name === "bler_rt_beam_web_quick.toml" ? "quick" : "custom";
    }
    if (serial !== state.rtConfigSerial) return;
    state.rtSelected[kind] = name;
    state.rtScenarioId = "";
    state.rtDirty = true;
    state.rtConfigPending = false;
    state.rtConfigPromise = null;
    $("rt-apply-toml").disabled = false;
    state.rtConfigError = "";
    populateRtPresetSelect();
    populateRtProfileSelects();
    renderRtSettings();
    await validateCurrentRun();
  } catch (error) {
    if (serial !== state.rtConfigSerial) return;
    state.rtConfigPending = false;
    state.rtConfigPromise = null;
    $("rt-apply-toml").disabled = false;
    state.rtConfigError = previousError;
    showAlert(error.message);
    populateRtProfileSelects();
    if (state.backend === "rt") {
      try { await validateCurrentRun(); }
      catch (validationError) {
        state.validation = null;
        renderCompatibility({ valid: false, errors: [{ kind: "validation", message: validationError.message }] });
      }
    }
  }
}

async function changeRtWaveform(value) {
  const name = value === "cp" ? "pusch_rt_4ue_cp.toml" : "pusch_4ue.toml";
  if (!state.configs.tx.includes(name)) {
    $("rt-waveform").value = state.rtWaveform;
    showAlert(`找不到波形配置 ${name}。`);
    return;
  }
  await changeRtProfile("tx", name);
  state.rtWaveform = state.rtSelected.tx === name ? value : (state.rtSelected.tx === "pusch_rt_4ue_cp.toml" ? "cp" : "dft");
  $("rt-waveform").value = state.rtWaveform;
  state.rtScenarioId = "";
  populateRtPresetSelect();
}

async function changeRtBudget(value) {
  const previous = state.rtBudget;
  if (!["quick", "long"].includes(value)) {
    $("rt-budget").value = previous;
    return;
  }
  if (value === "long" && (!cudaAvailable() || !state.configs.simulation.includes("bler_rt_beam.toml"))) {
    $("rt-budget").value = previous;
    showAlert("较长统计需要可用 CUDA 和 bler_rt_beam.toml；未自动更改设备或预算。");
    return;
  }
  const name = value === "long" ? "bler_rt_beam.toml" : "bler_rt_beam_web_quick.toml";
  if (!state.configs.simulation.includes(name)) {
    $("rt-budget").value = previous;
    showAlert(`找不到仿真预算配置 ${name}。`);
    return;
  }
  await changeRtProfile("simulation", name);
  state.rtBudget = state.rtSelected.simulation === name ? value : previous;
  $("rt-budget").value = state.rtBudget;
  state.rtScenarioId = "";
  populateRtPresetSelect();
}

function onRtPayloadInput(kind, event) {
  if (kind === "tx") state.rtTxText = event.target.value;
  else state.rtSimulationText = event.target.value;
  state.rtScenarioId = "";
  state.rtDirty = true;
  populateRtPresetSelect();
  updateRtSummary();
  scheduleValidation(false);
}

async function saveRtPayloadConfig(kind) {
  const name = state.rtSelected[kind];
  const text = kind === "tx" ? state.rtTxText : state.rtSimulationText;
  const button = $(kind === "tx" ? "rt-save-tx-config" : "rt-save-simulation-config");
  if (!name) return showAlert(`未选择 ${kind} profile。`);
  button.disabled = true;
  try {
    const saved = await api(
      `/api/configs/${encodeURIComponent(kind)}/${encodeURIComponent(name)}`,
      { method: "PUT", body: JSON.stringify({ text }) },
    );
    if (kind === "tx") state.rtTxText = saved.text;
    else state.rtSimulationText = saved.text;
    state.rtScenarioId = "";
    state.rtDirty = true;
    renderRtSettings();
    await validateCurrentRun();
    showAlert(`已保存 ${name}`, false);
  } catch (error) {
    showAlert(error.message);
  } finally {
    button.disabled = false;
  }
}

async function restoreRtPreset() {
  const scenarioId = state.rtPresetId || "rt-los-quick";
  await loadRtScenario(scenarioId, false);
}

async function uploadRtScene(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".zip")) {
    showAlert("场景包必须是 ZIP 文件。");
    event.target.value = "";
    return;
  }
  if (file.size > 8 * 1024 * 1024) {
    showAlert("场景包超过 8 MiB；请选择符合上限的 ZIP 文件。");
    event.target.value = "";
    return;
  }
  state.rtUploadPending = true;
  state.validation = null;
  renderCompatibility(null);
  $("rt-upload-trigger").disabled = true;
  try {
    const contentBase64 = await fileToBase64(file);
    const imported = await api("/api/rt/scenes", {
      method: "POST",
      body: JSON.stringify({ filename: file.name, content_base64: contentBase64 }),
    });
    state.rtSceneId = imported.scene_id;
    state.rtSceneFilename = file.name;
    state.rtSceneBundleHash = imported.bundle_sha256 || "";
    state.rtMeshSummary = imported.mesh_summary || {};
    const settings = cloneValue(state.rtDraftSettings || state.rtSettings);
    settings.rt.scene = "custom";
    settings.rt.scene_file = "scene.xml";
    delete settings.geometry;
    state.rtDraftSettings = settings;
    state.rtScenarioId = "";
    state.rtDirty = true;
    const serial = ++state.rtConfigSerial;
    state.rtConfigPending = true;
    state.rtConfigError = "";
    state.rtConfigPromise = sendRtSettings(serial, settings);
    const applied = await state.rtConfigPromise;
    if (!applied && serial === state.rtConfigSerial) throw new Error(state.rtConfigError || "场景包已上传，但 RT 场景配置未能应用。");
    $("rt-scene-source").value = "imported";
    $("rt-scene-kind").value = "custom";
    renderRtSettings();
    populateRtPresetSelect();
  } catch (error) {
    showAlert(error.message);
  } finally {
    state.rtUploadPending = false;
    $("rt-upload-trigger").disabled = false;
    event.target.value = "";
    if (state.backend === "rt" && state.validation) renderCompatibility(state.validation);
  }
}

async function changeRtSourceProfile(kind, event) {
  await changeRtProfile(kind, event.target.value);
}

async function activateRt() {
  await switchBackend("rt");
}

function activateCdl() {
  switchBackend("cdl").catch((error) => showAlert(error.message));
}

function wireRtUi() {
  $("backend-rt-button").addEventListener("click", () => activateRt().catch((error) => showAlert(error.message)));
  $("backend-cdl-button").addEventListener("click", activateCdl);
  $("rt-preset-select").addEventListener("change", (event) => {
    if (event.target.value === "custom") {
      state.rtScenarioId = "";
      state.rtDirty = true;
      populateRtPresetSelect();
      if (state.backend === "rt") validateCurrentRun().catch((error) => {
        state.validation = null;
        renderCompatibility({ valid: false, errors: [{ kind: "validation", message: error.message }] });
      });
      return;
    }
    loadRtScenario(event.target.value).catch((error) => showAlert(error.message));
  });
  $("rt-reset-preset").addEventListener("click", () => restoreRtPreset().catch((error) => showAlert(error.message)));
  $("rt-waveform").addEventListener("change", (event) => changeRtWaveform(event.target.value).catch((error) => showAlert(error.message)));
  $("rt-budget").addEventListener("change", (event) => changeRtBudget(event.target.value).catch((error) => showAlert(error.message)));
  $("rt-phy-device").addEventListener("change", (event) => setRtPhyDevice(event.target.value).catch((error) => showAlert(error.message)));
  $("rt-fields").addEventListener("input", onRtFieldInput);
  $("rt-fields").addEventListener("change", (event) => {
    if (event.target.matches("[data-rt-path]") && event.target.tagName === "SELECT") onRtFieldInput(event);
  });
  $("rt-scene-source").addEventListener("change", onRtSceneSourceChange);
  $("rt-scene-kind").addEventListener("change", onRtSceneKindChange);
  $("rt-profile-select").addEventListener("change", (event) => changeRtSourceProfile("rt", event).catch((error) => showAlert(error.message)));
  $("rt-tx-profile-select").addEventListener("change", (event) => changeRtSourceProfile("tx", event).catch((error) => showAlert(error.message)));
  $("rt-simulation-profile-select").addEventListener("change", (event) => changeRtSourceProfile("simulation", event).catch((error) => showAlert(error.message)));
  $("rt-expert-editor").addEventListener("input", () => {
    const changed = $("rt-expert-editor").value !== state.rtText;
    const wasDirty = state.rtExpertDirty;
    state.rtExpertDirty = changed;
    if (state.rtConfigPending || (changed && !wasDirty)) {
      clearTimeout(state.rtConfigTimer);
      state.rtConfigTimer = null;
      ++state.rtConfigSerial;
      state.rtConfigPending = false;
      state.rtConfigPromise = null;
      $("rt-apply-toml").disabled = false;
    }
    if (changed && !wasDirty) {
      state.rtDirty = true;
      state.rtScenarioId = "";
      populateRtPresetSelect();
    }
    $("rt-expert-status").textContent = changed ? "TOML 有未应用的修改；运行已锁定。" : "表单参数为当前有效配置。";
    if (changed) {
      state.validation = null;
      renderCompatibility({ valid: false, errors: [{ kind: "rt", message: "RT TOML 尚未应用。" }] });
    } else if (wasDirty) {
      state.rtConfigError = "";
      $("rt-config-error").classList.add("hidden");
      if (state.backend === "rt") validateCurrentRun().catch((error) => {
        state.validation = null;
        renderCompatibility({ valid: false, errors: [{ kind: "validation", message: error.message }] });
      });
    }
  });
  $("rt-tx-editor").addEventListener("input", (event) => onRtPayloadInput("tx", event));
  $("rt-simulation-editor").addEventListener("input", (event) => onRtPayloadInput("simulation", event));
  $("rt-save-tx-config").addEventListener("click", () => saveRtPayloadConfig("tx"));
  $("rt-save-simulation-config").addEventListener("click", () => saveRtPayloadConfig("simulation"));
  $("rt-apply-toml").addEventListener("click", applyRtToml);
  $("rt-upload-trigger").addEventListener("click", () => $("rt-scene-zip").click());
  $("rt-scene-zip").addEventListener("change", uploadRtScene);
  $("rt-user-filter").addEventListener("change", (event) => {
    state.rtResultUser = event.target.value;
    if (state.currentJob) renderResults(state.currentJob.points || [], state.currentJob);
  });
}

function formatMetric(value, digits = 4) {
  if (value === null || value === undefined || (typeof value === "string" && !value.trim())) return "—";
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

function finiteMetric(...values) {
  for (const value of values) {
    if (value === null || value === undefined || (typeof value === "string" && !value.trim())) continue;
    const number = Number(value);
    if (Number.isFinite(number)) return number;
  }
  return null;
}

function rtPointUser(point) {
  const value = point.user ?? point.user_name;
  if (typeof value === "string" && value) return value.toLowerCase();
  const index = finiteMetric(value, point.user_index, point.user_idx, point.user_id);
  return index === null ? "all" : `ue${index}`;
}

function renderResults(allPoints, job = state.currentJob) {
  const source = Array.isArray(allPoints) ? allPoints : [];
  const rt = state.resultBackend === "rt" || job?.channel_backend === "rt" || job?.backend === "rt" || Boolean(job?.config_names?.rt);
  const isSkippedPoint = (point) => Boolean(point?.skipped) || point?.status === "skipped";
  const visibleRows = source.filter((point) => point && (rt
    ? (state.rtResultUser === "all"
      ? rtPointUser(point) === "all"
      : rtPointUser(point) === state.rtResultUser)
    : !isSkippedPoint(point)));
  const measuredPoints = visibleRows.filter((point) => !isSkippedPoint(point));
  const skippedUser = rt ? (state.rtResultUser === "all" ? "all" : state.rtResultUser) : null;
  const skippedCount = rt
    ? source.filter((point) => isSkippedPoint(point) && rtPointUser(point) === skippedUser).length
    : source.filter((point) => point && point.skipped).length;
  const hasPoints = visibleRows.length > 0;
  $("rt-user-filter-wrap").classList.toggle("hidden", !rt);
  $("rt-user-filter").value = state.rtResultUser;
  $("results-empty").classList.toggle("hidden", hasPoints);
  $("results-content").classList.toggle("hidden", !hasPoints);
  $("result-subtitle").textContent = hasPoints
    ? `${new Set(visibleRows.map(comboLabel)).size} 个估计器/检测器组合 · ${measuredPoints.length} 个测量点${skippedCount ? ` · 跳过 ${skippedCount} 个点` : ""}${rt ? " · 固定快照条件 BLER" : ""}`
    : "完成仿真后将在此显示结果";
  const body = $("metrics-body");
  body.replaceChildren();
  $("chart-legend").replaceChildren();
  if (!hasPoints) {
    $("table-count").textContent = "0 条记录";
    return;
  }
  [...visibleRows].sort((a, b) =>
    String(a.channel_estimator).localeCompare(String(b.channel_estimator))
    || String(a.detector).localeCompare(String(b.detector))
    || (rt ? rtPointUser(a).localeCompare(rtPointUser(b)) : 0)
    || Number(a.snr_db) - Number(b.snr_db)
  ).forEach((point) => {
    const row = document.createElement("tr");
    const actualSnr = finiteMetric(point.actual_snr_db, point.actual_snr);
    const referenceSnr = finiteMetric(point.reference_snr_db, point.reference_snr, point.snr_db);
    const low = finiteMetric(point.bler_ci95_low, point.ci95_low);
    const high = finiteMetric(point.bler_ci95_high, point.ci95_high);
    const ci = low === null && high === null ? "—" : `${low === null ? "—" : formatMetric(low)} – ${high === null ? "—" : formatMetric(high)}`;
    const status = point.status || (point.skipped ? "skipped" : "complete");
    const reason = point.reason ? `：${point.reason}` : "";
    const statusText = status === "skipped"
      ? `跳过${reason}`
      : ["infeasible_rank", "singular_noise_covariance"].includes(status)
        ? `不可行 (${status})${reason}` : `${status}${reason}`;
    const values = [
      rt ? rtPointUser(point) : "—",
      comboLabel(point),
      formatMetric(point.snr_db, 2),
      formatMetric(actualSnr, 2),
      formatMetric(referenceSnr, 2),
      formatMetric(point.bler),
      ci,
      formatMetric(point.ber),
      statusText,
      formatMetric(point.crc_fail_rate),
      point.frames ?? "—",
      point.transport_blocks ?? "—",
      point.block_errors ?? "—",
      finiteMetric(point.runtime_s) === null ? "—" : `${formatMetric(point.runtime_s, 2)} s`,
    ];
    values.forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      if (rt && index === 8 && ["infeasible_rank", "singular_noise_covariance"].includes(status)) cell.className = "rt-infeasible";
      if (rt && index === 5 && finiteMetric(point.bler) === 0 && finiteMetric(point.block_errors, point.block_error_count) === 0) {
        cell.title = high === null ? "0 错误；准入报告未提供 95% 上界" : `0 错误；95% 上界 ${formatMetric(high)}`;
      }
      row.append(cell);
    });
    body.append(row);
  });
  $("table-count").textContent = `${visibleRows.length} 条记录`;
  $("chart-description").textContent = rt
    ? "对数刻度；0 错误只显示 95% 上界箭头，不作为 BLER=1/TB 的观测值。"
    : "对数刻度；零误块以统计上界标记，不绘制伪观测值。";
  requestAnimationFrame(() => drawChart(measuredPoints, rt));
}

function drawChart(points, isRt = false) {
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
  const prepared = points.flatMap((point) => {
    if (!point || point.skipped || ["infeasible_rank", "singular_noise_covariance"].includes(point.status)) return [];
    if (point.status && !["complete", "completed", "ok"].includes(point.status)) return [];
    const snr = finiteMetric(point.reference_snr_db, point.reference_snr, point.snr_db);
    const bler = finiteMetric(point.bler);
    if (snr === null || bler === null || bler < 0 || bler > 1) return [];
    const zeroErrors = bler === 0 && finiteMetric(point.block_errors, point.block_error_count) === 0;
    const upper = finiteMetric(point.bler_ci95_high, point.ci95_high);
    if (bler === 0 && (!zeroErrors || upper === null || upper <= 0)) return [];
    return [{ point, snr, bler, zeroErrors, plotBler: zeroErrors ? upper : bler }];
  });
  ctx.clearRect(0, 0, width, height);
  const legend = $("chart-legend");
  legend.replaceChildren();
  if (!prepared.length) return;
  const snrs = prepared.map((item) => item.snr);
  let xMin = Math.min(...snrs), xMax = Math.max(...snrs);
  if (xMin === xMax) { xMin -= 1; xMax += 1; }
  const yMinLog = Math.floor(Math.log10(Math.min(...prepared.map((item) => item.plotBler), .001)));
  const yMaxLog = 0;
  const x = (value) => pad.left + ((value - xMin) / (xMax - xMin)) * plotW;
  const y = (value) => pad.top + ((yMaxLog - Math.log10(Math.max(value, 10 ** yMinLog))) / (yMaxLog - yMinLog || 1)) * plotH;
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
  for (let index = 0; index <= ticks; index++) {
    const value = xMin + ((xMax - xMin) * index) / ticks;
    const px = x(value);
    ctx.strokeStyle = "#edf0ed"; ctx.beginPath(); ctx.moveTo(px, pad.top); ctx.lineTo(px, height - pad.bottom); ctx.stroke();
    ctx.fillStyle = "#74807b"; ctx.fillText(formatMetric(value, 1), px, height - pad.bottom + 10);
  }
  ctx.fillStyle = "#52605a"; ctx.font = "11px sans-serif"; ctx.fillText(isRt ? "参考 SNR (dB)" : "SNR (dB)", pad.left + plotW / 2, height - 16);
  ctx.save(); ctx.translate(15, pad.top + plotH / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.fillText("BLER", 0, 0); ctx.restore();
  const groups = new Map();
  prepared.forEach((item) => {
    const name = comboLabel(item.point);
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name).push(item);
  });
  [...groups.entries()].forEach(([name, values], index) => {
    const color = seriesColors[index % seriesColors.length];
    values.sort((a, b) => a.snr - b.snr);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    let connected = false;
    values.forEach((item) => {
      const px = x(item.snr), py = y(item.plotBler);
      if (item.zeroErrors) {
        connected = false;
        return;
      }
      if (connected) ctx.lineTo(px, py);
      else ctx.moveTo(px, py);
      connected = true;
    });
    ctx.stroke();
    values.forEach((item) => {
      const point = item.point;
      const px = x(item.snr), py = y(item.plotBler);
      const low = finiteMetric(point.bler_ci95_low, point.ci95_low);
      const high = finiteMetric(point.bler_ci95_high, point.ci95_high);
      if (low !== null && high !== null && high > 0 && high >= low) {
        ctx.strokeStyle = color; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(px, y(high)); ctx.lineTo(px, y(Math.max(low, 10 ** yMinLog))); ctx.stroke();
      }
      if (item.zeroErrors) {
        ctx.strokeStyle = color; ctx.fillStyle = "#fffefa"; ctx.lineWidth = 1.7;
        ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(px, py + 5); ctx.lineTo(px, py + 14); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(px - 3, py + 10); ctx.lineTo(px, py + 14); ctx.lineTo(px + 3, py + 10); ctx.stroke();
        ctx.fillStyle = color; ctx.font = "8px sans-serif"; ctx.textAlign = "center"; ctx.fillText("0错误，上界", px, py + 16);
      } else {
        ctx.fillStyle = color; ctx.strokeStyle = color; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.arc(px, py, 3.5, 0, Math.PI * 2); ctx.fill();
      }
    });
    const item = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.style.background = color;
    const text = document.createElement("b");
    text.textContent = name;
    item.append(swatch, text);
    legend.append(item);
  });
}

wireRtUi();
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
  if (state.currentJob && state.currentJob.points?.length) renderResults(state.currentJob.points, state.currentJob);
  if (state.rtInitialized) drawRtPreview();
  if (state.rxResult) drawConstellation(state.rxResult.constellation);
});

initConfigs()
  .then(() => Promise.all([loadRuns(), initRxConfigs()]))
  .catch((error) => showAlert(error.message));

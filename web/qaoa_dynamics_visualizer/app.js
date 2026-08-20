"use strict";

// `presentation_scenes` is retained as a compatibility name; the v2 microscope
// uses live layer/stage navigation instead of scripted scenes.
const presentation_scenes = [];
const SVG_NS = "http://www.w3.org/2000/svg";
const XHTML_NS = "http://www.w3.org/1999/xhtml";
const M = window.QAOAMath;
if (!M) throw new Error("central_math_notation_layer_missing");
const COLORS = {
  text: "#eef8f7", muted: "#9aafb4", grid: "rgba(159,202,211,.16)",
  teal: "#3ee0c1", cyan: "#57c7ff", gold: "#ffc857", red: "#ff6680",
  green: "#7be495", purple: "#ad91ff", orange: "#ff9567", dark: "#061015",
};

const app = {
  catalog: null,
  config: null,
  summary: null,
  checkpoint: null,
  track: null,
  algorithm: "penalty_x",
  depth: 110,
  layer: 21,
  stage: "before_cost",
  selectedState: 10377,
  termFilter: "all",
  checkpointCache: new Map(),
  runCache: new Map(),
  trackCache: new Map(),
  requestToken: 0,
  autoplay: null,
  presentation: false,
  complexHits: [],
};
const dom = {};

function byId(id) { return document.getElementById(id); }
function cacheDom() {
  [
    "algorithmSelect", "depthSelect", "comparisonSelect", "parameterCount",
    "previousButton", "playButton", "pauseButton", "nextButton", "layerSlider",
    "layerOutput", "speedSelect", "checkpointLabel", "gammaValue", "betaValue",
    "validationBadge", "modeToggle", "replaySource", "layerEquation", "circuitWindow", "operationName",
    "operationFormula", "operationAngle", "mixerName", "operationTeaching",
    "normalizationBadge", "energyValue", "pFeasValue", "pOptValue", "pCondValue",
    "transitionHeadline", "transitionName", "deltaProbabilityLabel", "deltaPhaseLabel",
    "deltaEnergyLabel", "deltaFeasLabel", "deltaOptLabel", "deltaProbability", "deltaPhase",
    "deltaEnergy", "deltaFeas", "deltaOpt", "gainList", "lossList", "phaseList",
    "topNSelect", "stateFilter", "stateSearch", "stateScope", "stateTableBody",
    "complexCanvas", "routeSvg", "routeTopState", "showOptimal", "showProbabilities",
    "showTopRoute", "evolutionCanvas", "evolutionMarker", "parameterSource",
    "parameterCanvas", "parameterTableBody", "selectedDetails", "trackerCanvas",
    "trackSelectedButton", "histogramCanvas", "mixerSvg", "mixerExplanation",
    "hamiltonianCounts", "costFormula", "hamiltonianSearch", "hamiltonianTerms",
    "phaseExamples", "optimizerStatus", "optimizerDetails", "comparisonPanel",
    "comparisonContent", "tooltip", "loadingOverlay", "loadingMessage", "errorOverlay",
    "errorMessage",
  ].forEach((id) => { dom[id] = byId(id); });
}

function stageCaption(stage) {
  if (stage === "before_cost") return "Input state";
  if (stage === "after_cost") return "After cost evolution";
  return "After mixer evolution";
}

function appendText(target, value) { target.append(document.createTextNode(String(value))); }
function appendMath(target, latex, options = {}) { target.append(M.make(latex, options)); }
function replaceWithMath(target, latex, options = {}) { M.render(target, latex, options); }
function mathCell(row, latex, className = "") {
  const cell = document.createElement("td");
  if (className) cell.className = className;
  replaceWithMath(cell, latex);
  row.append(cell);
  return cell;
}
function htmlMath(namespace, latex, className = "") {
  const element = document.createElementNS(namespace, "div");
  element.className = className;
  replaceWithMath(element, latex);
  return element;
}

async function fetchJSON(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${url} returned HTTP ${response.status}`);
  return response.json();
}
function number(value, digits = 6) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const x = Number(value);
  if (x !== 0 && Math.abs(x) < 1e-4) return x.toExponential(3);
  return x.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
}
function setupCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(280, Math.round(rect.width));
  const height = Math.max(210, Math.round(rect.height));
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0); context.clearRect(0, 0, width, height);
  return { context, width, height };
}
function text(context, value, x, y, options = {}) {
  context.save(); context.fillStyle = options.color || COLORS.muted;
  context.font = options.font || "11px Inter, system-ui, sans-serif";
  context.textAlign = options.align || "left"; context.textBaseline = options.baseline || "alphabetic";
  context.fillText(String(value), x, y); context.restore();
}
function line(context, values, xFor, yFor, color, dashed = false, width = 2) {
  context.save(); context.strokeStyle = color; context.lineWidth = width;
  if (dashed) context.setLineDash([5, 4]); context.beginPath();
  values.forEach((value, index) => { const x = xFor(index); const y = yFor(value); if (index) context.lineTo(x, y); else context.moveTo(x, y); });
  context.stroke(); context.restore();
}
function svg(name, attributes = {}) {
  const element = document.createElementNS(SVG_NS, name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
  return element;
}

function initializeControls() {
  app.catalog.algorithms.forEach((algorithm) => {
    const option = document.createElement("option"); option.value = algorithm.id; option.textContent = algorithm.label; dom.algorithmSelect.append(option);
  });
  app.catalog.depths.forEach((depth) => {
    const option = document.createElement("option"); option.value = String(depth); option.textContent = `${depth} layer${depth === 1 ? "" : "s"}`; dom.depthSelect.append(option);
  });
  const defaults = app.catalog.default;
  app.algorithm = defaults.algorithm; app.depth = defaults.depth; app.layer = defaults.layer; app.stage = defaults.stage;
  app.selectedState = app.catalog.reference_states.optimal.basis_index;
  dom.algorithmSelect.value = app.algorithm; dom.depthSelect.value = String(app.depth);
}

function bindEvents() {
  dom.algorithmSelect.addEventListener("change", () => { app.algorithm = dom.algorithmSelect.value; loadRun(); });
  dom.depthSelect.addEventListener("change", () => { app.depth = Number(dom.depthSelect.value); app.layer = Math.min(app.layer, app.depth); loadRun(); });
  dom.comparisonSelect.addEventListener("change", () => { dom.comparisonPanel.classList.toggle("hidden", dom.comparisonSelect.value !== "compare"); renderComparison(); });
  dom.layerSlider.addEventListener("input", () => { app.layer = Number(dom.layerSlider.value); if (app.layer === 0) app.stage = "before_cost"; loadCheckpoint(); });
  document.querySelectorAll("[data-stage]").forEach((button) => button.addEventListener("click", () => {
    if (app.layer === 0) app.layer = 1;
    app.stage = button.dataset.stage; loadCheckpoint();
  }));
  dom.previousButton.addEventListener("click", previousCheckpoint);
  dom.nextButton.addEventListener("click", nextCheckpoint);
  dom.playButton.addEventListener("click", startAutoplay);
  dom.pauseButton.addEventListener("click", stopAutoplay);
  dom.speedSelect.addEventListener("change", () => {
    if (app.autoplay) { stopAutoplay(); startAutoplay(); }
  });
  dom.topNSelect.addEventListener("change", loadCheckpoint);
  dom.stateFilter.addEventListener("change", loadCheckpoint);
  let searchTimer = null;
  dom.stateSearch.addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(loadCheckpoint, 220); });
  [dom.showOptimal, dom.showProbabilities, dom.showTopRoute].forEach((input) => input.addEventListener("change", renderRoute));
  dom.modeToggle.addEventListener("click", togglePresentation);
  dom.trackSelectedButton.addEventListener("click", loadTrack);
  document.querySelectorAll("[data-reference-state]").forEach((button) => button.addEventListener("click", () => selectReference(button.dataset.referenceState)));
  document.querySelectorAll("[data-term-filter]").forEach((button) => button.addEventListener("click", () => {
    app.termFilter = button.dataset.termFilter;
    document.querySelectorAll("[data-term-filter]").forEach((item) => item.classList.toggle("active", item === button)); renderHamiltonian();
  }));
  dom.hamiltonianSearch.addEventListener("input", renderHamiltonian);
  dom.complexCanvas.addEventListener("click", (event) => {
    const rect = dom.complexCanvas.getBoundingClientRect(); const x = event.clientX - rect.left; const y = event.clientY - rect.top;
    let best = null; let distance = 14 ** 2;
    app.complexHits.forEach((hit) => { const d = (hit.x - x) ** 2 + (hit.y - y) ** 2; if (d < distance) { distance = d; best = hit; } });
    if (best) selectState(best.index);
  });
  window.addEventListener("keydown", handleKeyboard);
  let resize = null;
  window.addEventListener("resize", () => { clearTimeout(resize); resize = setTimeout(renderCanvases, 120); });
}

function handleKeyboard(event) {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  if (event.key === " ") { event.preventDefault(); app.autoplay ? stopAutoplay() : startAutoplay(); }
  else if (event.key === "ArrowLeft") previousCheckpoint();
  else if (event.key === "ArrowRight") nextCheckpoint();
  else if (event.key === "ArrowUp") { event.preventDefault(); app.layer = Math.max(0, app.layer - 1); loadCheckpoint(); }
  else if (event.key === "ArrowDown") { event.preventDefault(); app.layer = Math.min(app.depth, app.layer + 1); loadCheckpoint(); }
  else if (event.key.toLowerCase() === "c") { if (app.layer === 0) app.layer = 1; app.stage = "after_cost"; loadCheckpoint(); }
  else if (event.key.toLowerCase() === "m") { if (app.layer === 0) app.layer = 1; app.stage = "after_mixer"; loadCheckpoint(); }
  else if (event.key === "1") { app.algorithm = "penalty_x"; dom.algorithmSelect.value = app.algorithm; loadRun(); }
  else if (event.key === "2") { app.algorithm = "global_grover"; dom.algorithmSelect.value = app.algorithm; loadRun(); }
  else if (event.key === "Escape" && app.presentation) togglePresentation();
}

async function loadRun() {
  stopAutoplay();
  app.track = null;
  const key = `${app.algorithm}:p${app.depth}`;
  const algorithmLabel = app.catalog.algorithms.find((item) => item.id === app.algorithm)?.label || "selected QAOA method";
  dom.loadingOverlay.classList.remove("hidden"); dom.loadingMessage.textContent = `Deterministically replaying ${algorithmLabel} at depth ${app.depth} from frozen angles…`;
  try {
    if (!app.runCache.has(key)) {
      const [config, summary] = await Promise.all([
        fetchJSON(`/api/run/${app.algorithm}/${app.depth}`),
        fetchJSON(`/api/evolution/${app.algorithm}/${app.depth}/summary`),
      ]);
      app.runCache.set(key, { config, summary });
    }
    ({ config: app.config, summary: app.summary } = app.runCache.get(key));
    app.layer = Math.min(app.layer, app.depth); dom.layerSlider.max = String(app.depth);
    renderRunStatic(); await loadCheckpoint(true); await loadTrack();
  } catch (error) { fatal(error); }
  finally { dom.loadingOverlay.classList.add("hidden"); }
}

function checkpointURL(algorithm = app.algorithm) {
  const parameters = new URLSearchParams({
    layer: String(app.layer), stage: app.stage, top: dom.topNSelect.value,
    filter: dom.stateFilter.value, selected: String(app.selectedState),
  });
  if (dom.stateSearch.value.trim()) parameters.set("q", dom.stateSearch.value.trim());
  return `/api/evolution/${algorithm}/${app.depth}/checkpoint?${parameters}`;
}

async function loadCheckpoint(initial = false) {
  if (!app.config) return;
  const token = ++app.requestToken; const key = checkpointURL();
  try {
    if (!app.checkpointCache.has(key)) app.checkpointCache.set(key, await fetchJSON(key));
    if (token !== app.requestToken) return;
    app.checkpoint = app.checkpointCache.get(key); renderAll();
    if (dom.comparisonSelect.value === "compare") await renderComparison();
  } catch (error) { if (!initial) fatal(error); else throw error; }
}

async function loadTrack() {
  if (!app.summary) return;
  const key = `${app.algorithm}:p${app.depth}:state${app.selectedState}`;
  if (!app.trackCache.has(key)) app.trackCache.set(key, await fetchJSON(`/api/evolution/${app.algorithm}/${app.depth}/track/${app.selectedState}`));
  app.track = app.trackCache.get(key); renderTracker();
}

function previousCheckpoint() {
  if (app.layer === 0) return;
  if (app.stage === "after_mixer") app.stage = "after_cost";
  else if (app.stage === "after_cost") app.stage = "before_cost";
  else if (app.layer === 1) { app.layer = 0; app.stage = "before_cost"; }
  else { app.layer -= 1; app.stage = "after_mixer"; }
  loadCheckpoint();
}
function nextCheckpoint() {
  if (app.layer === 0) { app.layer = 1; app.stage = "before_cost"; }
  else if (app.stage === "before_cost") app.stage = "after_cost";
  else if (app.stage === "after_cost") app.stage = "after_mixer";
  else if (app.layer < app.depth) { app.layer += 1; app.stage = "before_cost"; }
  else { stopAutoplay(); return; }
  loadCheckpoint();
}
function startAutoplay() {
  if (app.autoplay) return;
  dom.playButton.disabled = true; dom.pauseButton.disabled = false;
  app.autoplay = setInterval(nextCheckpoint, Number(dom.speedSelect.value));
}
function stopAutoplay() {
  if (app.autoplay) clearInterval(app.autoplay); app.autoplay = null;
  if (dom.playButton) { dom.playButton.disabled = false; dom.pauseButton.disabled = true; }
}

function renderRunStatic() {
  replaceWithMath(dom.parameterCount, String.raw`p=${app.depth}\Longrightarrow N_{\mathrm{param}}=${app.config.parameter_count}`);
  dom.parameterSource.replaceChildren(); appendText(dom.parameterSource, `Frozen ${app.config.algorithm_label} · `); appendMath(dom.parameterSource, `p=${app.depth}`); appendText(dom.parameterSource, ` · ${app.config.parameter_count} angles`);
  dom.optimizerStatus.textContent = app.config.optimizer.status;
  dom.optimizerDetails.replaceChildren();
  const optimizerItems = [
    [app.config.optimizer.name, "derivative-free classical optimizer", false],
    [`N_{\\mathrm{param}}=${app.config.parameter_count}=2(${app.depth})`, "variational parameters", true],
    [`${app.config.optimizer.evaluations} evaluations of a ${app.config.optimizer.evaluation_budget}-evaluation budget`, "frozen optimization record", false],
    [`${number(app.config.optimizer.runtime_seconds, 2)} s`, "recorded optimizer time", false],
  ];
  optimizerItems.forEach(([value, caption, isMath]) => { const item=document.createElement("div"); const strong=document.createElement("b"); if (isMath) replaceWithMath(strong,value); else strong.textContent=value; item.append(strong,document.createElement("br"),document.createTextNode(caption)); dom.optimizerDetails.append(item); });
  renderParameterTable(); renderMixer(); renderHamiltonian();
}

function renderAll() {
  renderController(); renderCircuit(); renderOperation(); renderMetrics(); renderTransition();
  renderStateTable(); drawComplex(); renderRoute(); drawEvolution(); drawParameters();
  renderSelected(); renderTracker(); drawHistogram(); renderHamiltonian();
}
function renderCanvases() {
  if (!app.checkpoint) return;
  drawComplex(); drawEvolution(); drawParameters(); renderTracker(); drawHistogram();
}

function renderController() {
  dom.algorithmSelect.value = app.algorithm; dom.depthSelect.value = String(app.depth);
  dom.layerSlider.value = String(app.layer); replaceWithMath(dom.layerOutput, String.raw`\ell=${app.layer}\,/\,p=${app.depth}`);
  document.querySelectorAll("[data-stage]").forEach((button) => {
    button.classList.toggle("active", app.layer > 0 && button.dataset.stage === app.stage);
    button.disabled = app.layer === 0;
    button.replaceChildren(); const label=document.createElement("span"); label.textContent=stageCaption(button.dataset.stage); const state=document.createElement("small"); replaceWithMath(state,M.stageState(Math.max(1,app.layer),button.dataset.stage)); button.append(label,state);
  });
  const params = app.layer > 0 ? app.config.parameters[app.layer - 1] : null;
  dom.checkpointLabel.replaceChildren(); appendText(dom.checkpointLabel, app.layer === 0 ? "Initial state · " : `Layer ${app.layer} · ${stageCaption(app.stage)} · `); appendMath(dom.checkpointLabel,M.stageState(app.layer,app.stage));
  replaceWithMath(dom.gammaValue, params ? M.instantiatedGamma(app.layer,params.gamma) : String.raw`\gamma_\ell=\text{--}`);
  replaceWithMath(dom.betaValue, params ? M.instantiatedBeta(app.layer,params.beta) : String.raw`\beta_\ell=\text{--}`);
  dom.replaySource.textContent = `deterministic replay · ${app.checkpoint.checkpoint_index + 1}/${1 + 2 * app.depth} stored checkpoints in memory`;
  replaceWithMath(dom.parameterCount, String.raw`p=${app.depth}\Longrightarrow N_{\mathrm{param}}=${2 * app.depth}`);
}

function renderCircuit() {
  replaceWithMath(dom.layerEquation, app.layer === 0 ? M.FORMULAS.fullEvolution : M.currentLayerEvolution(app.layer,app.algorithm), { displayMode: true });
  dom.circuitWindow.replaceChildren();
  const currentIndex = app.checkpoint.checkpoint_index; const start = Math.max(1, app.layer - 2); const end = Math.min(app.depth, Math.max(3, app.layer + 2));
  const nodes = [];
  if (start === 1) nodes.push({ label: M.stageState(0,"before_cost"), detail: String.raw`\lvert+\rangle^{\otimes14}`, index: 0, math: true }); else nodes.push({ label: "…", detail: `layers 1–${start - 1}`, ellipsis: true });
  for (let layer = start; layer <= end; layer += 1) {
    if (layer === app.layer && app.stage === "before_cost") nodes.push({ label: M.stageState(layer,"before_cost"), detail: "INPUT STATE", index: 2 * (layer - 1), input: true, math: true });
    nodes.push({ label: M.costUnitary(layer), detail: M.instantiatedGamma(layer,app.config.parameters[layer - 1].gamma), index: 2 * layer - 1, math: true });
    nodes.push({ label: M.mixerUnitary(layer,app.algorithm), detail: M.instantiatedBeta(layer,app.config.parameters[layer - 1].beta), index: 2 * layer, math: true });
  }
  if (end < app.depth) nodes.push({ label: "…", detail: `to layer ${app.depth}`, ellipsis: true }); else nodes.push({ label: "Measure", detail: "routes", index: 2 * app.depth + 1 });
  nodes.forEach((node) => {
    const item = document.createElement("div"); item.className = "circuit-node";
    if (node.ellipsis) item.classList.add("ellipsis");
    else if (
      (app.layer > 0 && app.stage === "before_cost" && node.input)
      || (app.stage !== "before_cost" && node.index === currentIndex)
      || (app.layer === 0 && node.index === 0)
    ) item.classList.add("current");
    else if (node.index <= currentIndex) item.classList.add("past"); else item.classList.add("future");
    const label=document.createElement("b"), detail=document.createElement("span"); if (node.math) { replaceWithMath(label,node.label); if (node.detail.startsWith("\\")) replaceWithMath(detail,node.detail); else detail.textContent=node.detail; } else { label.textContent=node.label; detail.textContent=node.detail; } item.append(label,detail); dom.circuitWindow.append(item);
  });
  dom.circuitWindow.querySelector(".current")?.scrollIntoView({ inline: "center", block: "nearest" });
}

function renderOperation() {
  const current = app.checkpoint.current_operation;
  dom.mixerName.textContent = `Mixer = ${app.config.algorithm_label}`;
  if (app.layer === 0 || app.stage === "before_cost") {
    dom.operationName.textContent = app.layer === 0 ? "Initial state" : "Layer input";
    replaceWithMath(dom.operationFormula, app.layer === 0 ? String.raw`\lvert\psi_0\rangle=\lvert+\rangle^{\otimes14}` : M.stageState(app.layer,"before_cost"));
    dom.operationAngle.textContent = "No unitary applied at this checkpoint";
    dom.operationTeaching.textContent = app.layer === 0 ? "Uniform amplitude over all 16,384 edge bitstrings." : "This state is exactly the previous layer's after-mixer state.";
  } else if (app.stage === "after_cost") {
    dom.operationName.textContent = "Cost evolution"; replaceWithMath(dom.operationFormula,M.instantiatedCostUnitary(app.layer,current.gamma));
    replaceWithMath(dom.operationAngle,M.instantiatedGamma(app.layer,current.gamma));
    dom.operationTeaching.textContent = "Energy is encoded into relative phase. Computational-basis probability is unchanged.";
  } else {
    dom.operationName.textContent = `${app.config.algorithm_label} mixer evolution`;
    replaceWithMath(dom.operationFormula,M.instantiatedMixerUnitary(app.layer,current.beta,app.algorithm)); replaceWithMath(dom.operationAngle,M.instantiatedBeta(app.layer,current.beta));
    dom.operationTeaching.textContent = "Phase-tagged amplitudes interfere, changing the probability distribution.";
  }
}

function renderMetrics() {
  const metrics = app.checkpoint.metrics;
  replaceWithMath(dom.energyValue,M.numeric(metrics.expected_hc,6)); replaceWithMath(dom.pFeasValue,M.numeric(metrics.p_feas,6));
  replaceWithMath(dom.pOptValue,M.numeric(metrics.p_opt,6)); replaceWithMath(dom.pCondValue,M.numeric(metrics.p_opt_given_feasible,6));
  const normError = app.checkpoint.validation.maxima.normalization_error;
  replaceWithMath(dom.normalizationBadge,String.raw`\sum_x\lvert a_x\rvert^2=1,\quad\left\lvert1-\sum_x\lvert a_x\rvert^2\right\rvert=${M.numeric(normError,3)}`); dom.normalizationBadge.classList.add("valid");
  dom.transitionHeadline.className = "transition-headline";
  dom.transitionHeadline.replaceChildren();
  if (app.stage === "after_cost") { const bold=document.createElement("b"); bold.textContent="Probability is invariant within numerical tolerance; phase rotates."; dom.transitionHeadline.append(bold,document.createElement("br")); appendMath(dom.transitionHeadline,String.raw`a_x\mapsto a_xe^{-i\gamma_{${app.layer}}E_x},\quad\lvert a_x\rvert^2\mapsto\lvert a_x\rvert^2`); }
  else if (app.stage === "after_mixer") { dom.transitionHeadline.classList.add("mixer"); const bold=document.createElement("b"); bold.textContent="Interference redistributes amplitude and probability."; dom.transitionHeadline.append(bold,document.createElement("br")); appendMath(dom.transitionHeadline,String.raw`${M.stageState(app.layer,"after_cost")}\xrightarrow{\,${M.mixerUnitary(app.layer,app.algorithm)}\,}${M.stageState(app.layer,"after_mixer")}`); }
  else { appendText(dom.transitionHeadline,"No operation is applied at this input checkpoint: "); appendMath(dom.transitionHeadline,M.stageState(app.layer,app.stage)); }
}

function renderTransition() {
  const transition = app.checkpoint.transition; const beforeStage=app.stage === "after_mixer" ? "after_cost" : app.stage === "after_cost" ? "before_cost" : app.stage;
  replaceWithMath(dom.transitionName,String.raw`${M.stageState(app.layer,beforeStage)}\;\longrightarrow\;${M.stageState(app.layer,app.stage)}`);
  const prefix=M.transitionPrefix(app.stage); const delta=prefix ? String.raw`\Delta_{${prefix}}` : String.raw`\Delta`;
  replaceWithMath(dom.deltaProbabilityLabel,String.raw`\max_x\lvert${delta}P(x)\rvert`); replaceWithMath(dom.deltaPhaseLabel,String.raw`\max_x\lvert${delta}\arg(a_x)\rvert`); replaceWithMath(dom.deltaEnergyLabel,String.raw`${delta}\langle H_C\rangle`); replaceWithMath(dom.deltaFeasLabel,String.raw`${delta}p_{\mathrm{feas}}`); replaceWithMath(dom.deltaOptLabel,String.raw`${delta}p_{\mathrm{opt}}`);
  replaceWithMath(dom.deltaProbability,M.numeric(transition.probability_max_abs_delta,5));
  replaceWithMath(dom.deltaPhase,String.raw`${M.numeric(transition.phase_max_abs_delta,5)}\,\mathrm{rad}`);
  replaceWithMath(dom.deltaEnergy,M.numeric(transition.metrics_delta.expected_hc,5,true)); replaceWithMath(dom.deltaFeas,M.numeric(transition.metrics_delta.p_feas,5,true)); replaceWithMath(dom.deltaOpt,M.numeric(transition.metrics_delta.p_opt,5,true));
  renderChangeList(dom.gainList, transition.top_probability_gains, "probability_delta");
  renderChangeList(dom.lossList, transition.top_probability_losses, "probability_delta");
  renderChangeList(dom.phaseList, transition.largest_phase_rotations, "phase_delta");
}
function renderChangeList(target, rows, field) {
  target.replaceChildren();
  if (!rows.length) { const li = document.createElement("li"); li.textContent = "no operation at this checkpoint"; target.append(li); return; }
  rows.forEach((row) => { const li = document.createElement("li"); appendMath(li,M.ket(row.bitstring)); appendText(li,"  "); const delta=M.transitionPrefix(app.stage) ? String.raw`\Delta_{${M.transitionPrefix(app.stage)}}` : String.raw`\Delta`; appendMath(li,field === "phase_delta" ? String.raw`${delta}\arg(a_x)=${M.numeric(row[field],5,true)}\,\mathrm{rad}` : String.raw`${delta}P(x)=${M.numeric(row[field],5,true)}`); target.append(li); });
}

function renderStateTable() {
  dom.stateTableBody.replaceChildren(); const stateData = app.checkpoint.states;
  dom.stateScope.textContent = `${stateData.returned_count} displayed states · ${stateData.matching_count.toLocaleString()} matching · Hilbert-space dimension ${stateData.full_dimension.toLocaleString()}`;
  stateData.rows.forEach((row) => {
    const tr = document.createElement("tr"); tr.classList.toggle("selected", row.basis_index === app.selectedState);
    const stateCell=mathCell(tr,M.ket(row.bitstring)); const index=document.createElement("small"); replaceWithMath(index,`x=${row.basis_index}`); stateCell.append(document.createElement("br"),index);
    mathCell(tr,M.numeric(row.probability,6)); mathCell(tr,M.numeric(row.magnitude,6)); mathCell(tr,M.numeric(row.real,6)); mathCell(tr,M.numeric(row.imag,6)); mathCell(tr,String.raw`${M.numeric(row.phase,5)}\,\mathrm{rad}`); mathCell(tr,M.numeric(row.energy,2));
    const feasible=document.createElement("td"); feasible.className=row.feasible?"yes":""; feasible.textContent=row.feasible?"✓":"—"; tr.append(feasible); const optimal=document.createElement("td"); optimal.className=row.optimal?"star":""; optimal.textContent=row.optimal?"★":"—"; tr.append(optimal); const route=document.createElement("td"); route.textContent=row.route_text||"invalid edge set"; tr.append(route);
    tr.addEventListener("click", () => selectState(row.basis_index)); dom.stateTableBody.append(tr);
  });
}

async function selectState(index) { app.selectedState = Number(index); await loadCheckpoint(); await loadTrack(); }
async function selectReference(name) { app.selectedState = app.catalog.reference_states[name].basis_index; await loadCheckpoint(); app.track = app.summary.tracked_reference_states[name]; app.trackCache.set(`${app.algorithm}:p${app.depth}:state${app.selectedState}`, app.track); renderTracker(); }

function drawComplex() {
  const { context, width, height } = setupCanvas(dom.complexCanvas); const rows = app.checkpoint.states.rows;
  const triplet = app.checkpoint.selected_stage_triplet; const all = rows.concat(Object.values(triplet));
  const extent = Math.max(1e-7, ...all.flatMap((row) => [Math.abs(row.real), Math.abs(row.imag)])) * 1.18;
  const cx = width / 2, cy = height / 2, radius = Math.min(width, height) * .41;
  context.strokeStyle = COLORS.grid; context.lineWidth = 1; context.beginPath(); context.moveTo(26, cy); context.lineTo(width - 20, cy); context.moveTo(cx, 18); context.lineTo(cx, height - 28); context.stroke();
  const point = (row) => ({ x: cx + row.real / extent * radius, y: cy - row.imag / extent * radius });
  app.complexHits = [];
  rows.forEach((row) => { const p = point(row); context.beginPath(); context.arc(p.x, p.y, row.basis_index === app.selectedState ? 6 : 3.5, 0, Math.PI * 2); context.fillStyle = row.optimal ? COLORS.gold : row.feasible ? COLORS.green : COLORS.red; context.globalAlpha = row.basis_index === app.selectedState ? 1 : .72; context.fill(); context.globalAlpha = 1; app.complexHits.push({ x: p.x, y: p.y, index: row.basis_index }); });
  const pathRows = [triplet.before_cost, triplet.after_cost, triplet.after_mixer]; const positions = pathRows.map(point);
  context.strokeStyle = COLORS.purple; context.lineWidth = 2.4; context.beginPath(); positions.forEach((p, index) => index ? context.lineTo(p.x, p.y) : context.moveTo(p.x, p.y)); context.stroke();
  positions.forEach((p, index) => { context.beginPath(); context.arc(p.x, p.y, 5.2, 0, Math.PI * 2); context.fillStyle = [COLORS.cyan, COLORS.gold, COLORS.teal][index]; context.fill(); });
}

function renderRoute() {
  const graph = app.checkpoint.graph; dom.routeSvg.replaceChildren();
  const defs = svg("defs"); const marker = svg("marker", { id: "arrow", markerWidth: 8, markerHeight: 8, refX: 7, refY: 3, orient: "auto", markerUnits: "strokeWidth" }); marker.append(svg("path", { d: "M0,0 L0,6 L7,3 z", fill: "context-stroke" })); defs.append(marker); dom.routeSvg.append(defs);
  const position = new Map(graph.nodes.map((node) => [node.id, { x: node.x * 730 + 15, y: node.y * 345 + 18 }]));
  graph.edges.forEach((edge) => {
    const a = position.get(edge.u), b = position.get(edge.v); const dx = b.x - a.x, dy = b.y - a.y, length = Math.hypot(dx, dy), ux = dx / length, uy = dy / length;
    const lineElement = svg("line", { x1: a.x + ux * 21, y1: a.y + uy * 21, x2: b.x - ux * 25, y2: b.y - uy * 25, class: "graph-edge", "stroke-width": 1.4 + 7.5 * edge.selection_probability, opacity: .28 + .72 * edge.selection_probability });
    if (dom.showOptimal.checked && edge.optimal) lineElement.classList.add("optimal"); if (dom.showTopRoute.checked && edge.top_route) lineElement.classList.add("top-route"); if (edge.selected_state) lineElement.setAttribute("stroke-dasharray", "6 3"); dom.routeSvg.append(lineElement);
    const labelX=(a.x+b.x)/2-uy*12, labelY=(a.y+b.y)/2+ux*12; const foreign=svg("foreignObject",{x:labelX-63,y:labelY-14,width:126,height:30,class:"edge-label-object"}); const label=htmlMath(XHTML_NS,dom.showProbabilities.checked?M.edgeMarginal(edge.u,edge.v,edge.selection_probability):`w_{(${edge.u},${edge.v})}=${edge.weight}`,"edge-math-label"); foreign.append(label); dom.routeSvg.append(foreign);
  });
  graph.nodes.forEach((node) => { const p = position.get(node.id); const circle = svg("circle", { cx: p.x, cy: p.y, r: 20, class: `graph-node ${node.id === graph.source ? "source" : ""} ${node.id === graph.target ? "target" : ""}` }); const label = svg("text", { x: p.x, y: p.y, class: "node-label" }); label.textContent = node.id; dom.routeSvg.append(circle, label); });
  const top = app.checkpoint.metrics.top_state_index; const state = app.checkpoint.states.rows.find((row) => row.basis_index === top);
  dom.routeTopState.replaceChildren(); appendMath(dom.routeTopState,M.FORMULAS.mostProbable); appendText(dom.routeTopState," · "); if (state) appendMath(dom.routeTopState,String.raw`x_{\max}=${M.ket(state.bitstring)}`); appendText(dom.routeTopState,state?.route_text?` · decoded route ${state.route_text}`:` · basis index ${top} is infeasible`);
}

function panelFrame(context, x, y, width, height) {
  context.strokeStyle = COLORS.grid; context.strokeRect(x, y, width, height); return { left: x + 48, right: x + width - 12, top: y + 28, bottom: y + height - 28 };
}
function drawEvolution() {
  const { context, width, height } = setupCanvas(dom.evolutionCanvas); const gap = 18, pw = (width - gap) / 2, ph = (height - gap) / 2;
  const specs = ["expected_hc", "p_feas", "p_opt", "p_opt_given_feasible"];
  specs.forEach((field, panelIndex) => {
    const ox = (panelIndex % 2) * (pw + gap), oy = Math.floor(panelIndex / 2) * (ph + gap); const plot = panelFrame(context, ox, oy, pw, ph);
    const cost = app.summary.layers.map((layer) => layer.after_cost[field] ?? 0); const mixer = app.summary.layers.map((layer) => layer.after_mixer[field] ?? 0); const all = cost.concat(mixer);
    let min = Math.min(...all), max = Math.max(...all); if (field !== "expected_hc") min = 0; if (max === min) max = min + 1;
    const xFor = (index) => plot.left + index / Math.max(1, app.depth - 1) * (plot.right - plot.left); const yFor = (value) => plot.bottom - (value - min) / (max - min) * (plot.bottom - plot.top);
    line(context, cost, xFor, yFor, COLORS.cyan, true, 1.5); line(context, mixer, xFor, yFor, COLORS.teal, false, 2);
    text(context, number(max, field === "expected_hc" ? 2 : 4), plot.left - 5, plot.top + 4, { align: "right" }); text(context, number(min, field === "expected_hc" ? 2 : 4), plot.left - 5, plot.bottom, { align: "right" });
    if (app.layer > 0) { const currentMetric = app.summary.layers[app.layer - 1][app.stage][field] ?? 0; const x = xFor(app.layer - 1), y = yFor(currentMetric); context.strokeStyle = COLORS.gold; context.beginPath(); context.moveTo(x, plot.top); context.lineTo(x, plot.bottom); context.stroke(); context.beginPath(); context.arc(x, y, 5, 0, Math.PI * 2); context.fillStyle = COLORS.gold; context.fill(); }
    text(context, "1", plot.left, plot.bottom + 17); text(context, String(app.depth), plot.right, plot.bottom + 17, { align: "right" });
  });
  dom.evolutionMarker.replaceChildren(); appendText(dom.evolutionMarker,"Current marker: "); appendMath(dom.evolutionMarker,String.raw`\ell=${app.layer},\;${M.stageState(app.layer,app.stage)}`); appendText(dom.evolutionMarker," · raw trajectories");
}

function renderParameterTable() {
  dom.parameterTableBody.replaceChildren(); app.config.parameters.forEach((row) => { const tr = document.createElement("tr"); tr.dataset.layer = row.layer; mathCell(tr,String.raw`\ell=${row.layer}`); mathCell(tr,`${M.gamma(row.layer)}=${M.numeric(row.gamma,10)}`); mathCell(tr,`${M.beta(row.layer)}=${M.numeric(row.beta,10)}`); tr.addEventListener("click", () => { app.layer = row.layer; app.stage = "before_cost"; loadCheckpoint(); }); dom.parameterTableBody.append(tr); });
}
function drawParameters() {
  const { context, width, height } = setupCanvas(dom.parameterCanvas); const rows = app.config.parameters, left = 45, right = width - 18, top = 22, bottom = height - 34; const max = Math.max(...rows.flatMap((row) => [row.gamma, row.beta]), 1e-8); const xFor = (index) => left + index / Math.max(1, rows.length - 1) * (right - left), yFor = (value) => bottom - value / max * (bottom - top);
  context.strokeStyle = COLORS.grid; context.strokeRect(left, top, right - left, bottom - top); line(context, rows.map((row) => row.gamma), xFor, yFor, COLORS.gold); line(context, rows.map((row) => row.beta), xFor, yFor, COLORS.purple);
  if (app.layer > 0) { const x = xFor(app.layer - 1); context.strokeStyle = COLORS.teal; context.beginPath(); context.moveTo(x, top); context.lineTo(x, bottom); context.stroke(); }
  dom.parameterTableBody.querySelectorAll("tr").forEach((row) => row.classList.toggle("current", Number(row.dataset.layer) === app.layer));
}

function renderSelected() {
  const row = app.checkpoint.selected_state; dom.selectedDetails.replaceChildren();
  const items=[
    [M.ket(row.bitstring),"Computational-basis state",true,false],
    [`x=${row.basis_index}`,"Basis index",true,false],
    [row.optimal?"OPTIMAL ★":row.feasible?"FEASIBLE":"INFEASIBLE","Classification",false,false],
    [`P(x)=${M.numeric(row.probability,6)}`,"Probability",true,false],
    [M.stateAmplitude(row.real,row.imag),"Complex amplitude",true,false],
    [String.raw`\arg(a_x)=${M.numeric(row.phase,6)}\,\mathrm{rad}`,"Phase",true,false],
    [`E_x=${M.numeric(row.energy,3)}`,"Cost-Hamiltonian energy",true,false],
    [`C(x)=${M.numeric(row.route_cost,2)}`,"Decoded route cost",true,false],
    [row.route_text||"not a valid route","Decoded route",false,true],
  ];
  items.forEach(([value,label,isMath,wide])=>{ const item=document.createElement("article"); if(wide)item.classList.add("wide"); const caption=document.createElement("span"); caption.textContent=label; const strong=document.createElement("strong"); if(isMath)replaceWithMath(strong,value); else strong.textContent=value; item.append(caption,strong); dom.selectedDetails.append(item); });
  const edgeItem=document.createElement("article"); edgeItem.className="wide"; const edgeCaption=document.createElement("span"); edgeCaption.textContent="Selected edge variables"; const edgeValue=document.createElement("strong"); replaceWithMath(edgeValue,String.raw`\{e_j:x_{e_j}=1\}=\{${row.selected_edge_indices.join(",")||String.raw`\varnothing`}\}`); edgeItem.append(edgeCaption,edgeValue); dom.selectedDetails.append(edgeItem);
}
function renderTracker() {
  if (!app.track) return; const { context, width, height } = setupCanvas(dom.trackerCanvas); const values = app.track.values, half = height / 2, left = 48, right = width - 16; const xFor = (index) => left + index / Math.max(1, values.length - 1) * (right - left);
  const panels = [{ key: "probability", color: COLORS.teal, min: 0, max: Math.max(...values.map((v) => v.probability), 1e-12), y0: 18, y1: half - 20 }, { key: "phase", color: COLORS.purple, min: -Math.PI, max: Math.PI, y0: half + 12, y1: height - 30 }];
  panels.forEach((panel) => { context.strokeStyle = COLORS.grid; context.strokeRect(left, panel.y0, right - left, panel.y1 - panel.y0); const yFor = (value) => panel.y1 - (value - panel.min) / (panel.max - panel.min) * (panel.y1 - panel.y0); line(context, values.map((value) => value[panel.key]), xFor, yFor, panel.color); const currentIndex = app.checkpoint.checkpoint_index; context.beginPath(); context.arc(xFor(currentIndex), yFor(values[currentIndex][panel.key]), 5, 0, Math.PI * 2); context.fillStyle = COLORS.gold; context.fill(); });
}

function drawHistogram() {
  const { context, width, height } = setupCanvas(dom.histogramCanvas); const rows = app.checkpoint.energy_histogram, left = 52, right = width - 18, top = 22, bottom = height - 38; const minE = rows[0].energy, maxE = rows.at(-1).energy; const floor = -12; const transform = (mass) => Math.max(floor, Math.log10(Math.max(1e-12, mass))); const xFor = (value) => left + (value - minE) / (maxE - minE) * (right - left), yFor = (value) => bottom - (value - floor) / -floor * (bottom - top);
  context.strokeStyle = COLORS.grid; context.strokeRect(left, top, right - left, bottom - top); line(context, rows.map((row) => transform(row.probability_mass)), (i) => xFor(rows[i].energy), yFor, COLORS.cyan); line(context, rows.map((row) => transform(row.feasible_probability_mass)), (i) => xFor(rows[i].energy), yFor, COLORS.green);
  text(context, String(minE), left, bottom + 18); text(context, String(maxE), right, bottom + 18, { align: "right" });
}

function renderMixer() {
  const mixer = app.config.mixer; dom.mixerSvg.replaceChildren();
  if (app.algorithm === "penalty_x") {
    const points = [[110,80,"000"],[270,55,"001"],[270,145,"010"],[110,175,"100"],[430,90,"011"],[430,185,"110"]]; const links = [[0,1],[0,2],[0,3],[1,4],[2,4],[2,5],[3,5]];
    links.forEach(([a,b]) => dom.mixerSvg.append(svg("line", { x1: points[a][0], y1: points[a][1], x2: points[b][0], y2: points[b][1], stroke: COLORS.teal, "stroke-width": 2, opacity: .55 })));
    points.forEach(([x,y,label]) => { const c = svg("circle", { cx:x, cy:y, r:24, fill:"#142a34", stroke:COLORS.teal, "stroke-width":2 }); const foreign=svg("foreignObject",{x:x-31,y:y-12,width:62,height:24}); foreign.append(htmlMath(XHTML_NS,M.ket(label),"mixer-state-label")); dom.mixerSvg.append(c,foreign); });
    const caption=svg("text",{x:310,y:265,"text-anchor":"middle",fill:COLORS.muted,"font-size":13}); caption.textContent="Each edge changes one bit; the actual graph is the 14-dimensional hypercube."; dom.mixerSvg.append(caption);
  } else {
    const center = [310,145]; for (let i=0;i<12;i+=1) { const angle=2*Math.PI*i/12, x=center[0]+190*Math.cos(angle), y=center[1]+95*Math.sin(angle); dom.mixerSvg.append(svg("line",{x1:center[0],y1:center[1],x2:x,y2:y,stroke:COLORS.purple,"stroke-width":1.5,opacity:.45})); dom.mixerSvg.append(svg("circle",{cx:x,cy:y,r:9,fill:"#142a34",stroke:COLORS.purple})); }
    dom.mixerSvg.append(svg("circle",{cx:center[0],cy:center[1],r:42,fill:"rgba(173,145,255,.14)",stroke:COLORS.purple,"stroke-width":3})); const foreign=svg("foreignObject",{x:center[0]-45,y:center[1]-15,width:90,height:30}); foreign.append(htmlMath(XHTML_NS,"\\Pi_s","mixer-projector-label")); dom.mixerSvg.append(foreign); const caption=svg("text",{x:310,y:285,"text-anchor":"middle",fill:COLORS.muted,"font-size":13}); caption.textContent="Rank-one global redistribution relative to the uniform full-space reference state."; dom.mixerSvg.append(caption);
  }
  dom.mixerExplanation.replaceChildren(); const formula=document.createElement("div"); replaceWithMath(formula,app.algorithm==="penalty_x"?M.FORMULAS.xMixer:M.FORMULAS.groverMixer,{displayMode:true}); dom.mixerExplanation.append(formula); if(app.algorithm==="penalty_x") { const detail=document.createElement("div"); replaceWithMath(detail,M.FORMULAS.xLocal); dom.mixerExplanation.append(detail); } else { const state=document.createElement("div"); replaceWithMath(state,M.FORMULAS.uniformState); dom.mixerExplanation.append(state); } const geometry=document.createElement("p"); geometry.textContent=mixer.geometry; const boundary=document.createElement("p"); const bold=document.createElement("b"); bold.textContent="Global-Grover QAOA is not standard Grover search."; boundary.append(bold,document.createTextNode(" No oracle directly marks the optimal route in this experiment.")); dom.mixerExplanation.append(geometry,boundary);
}

function renderHamiltonian() {
  if (!app.catalog) return; const cost = app.catalog.hamiltonian.cost; dom.hamiltonianCounts.replaceChildren(); appendMath(dom.hamiltonianCounts,String.raw`${cost.counts.constant}I+${cost.counts.linear}Z_j+${cost.counts.zz}Z_jZ_k`); appendText(dom.hamiltonianCounts," · nonzero stored terms");
  const gamma = app.layer > 0 ? app.config.parameters[app.layer - 1].gamma : null;
  dom.costFormula.replaceChildren(); const hc=document.createElement("div"); replaceWithMath(hc,M.FORMULAS.costHamiltonian,{displayMode:true}); const c0=document.createElement("div"); replaceWithMath(c0,String.raw`c_0=${M.numeric(cost.constant,6)}`); const mapping=document.createElement("div"); replaceWithMath(mapping,String.raw`${M.FORMULAS.routingQubo},\qquad ${M.FORMULAS.binaryToIsing}`); dom.costFormula.append(hc,c0,mapping); if(gamma===null) appendText(dom.costFormula,"Select a layer to instantiate the cost unitary."); else { const current=document.createElement("div"); replaceWithMath(current,M.instantiatedCostUnitary(app.layer,gamma)); dom.costFormula.append(current); }
  const query = dom.hamiltonianSearch.value.trim().toLowerCase(); let terms = [{ kind:"constant", label:"I", coefficient:cost.constant }].concat(cost.linear, cost.couplings);
  if (app.termFilter === "linear") terms = terms.filter((term) => term.kind === "Z"); else if (app.termFilter === "zz") terms = terms.filter((term) => term.kind === "ZZ");
  if (query) terms = terms.filter((term) => term.label.toLowerCase().includes(query)); dom.hamiltonianTerms.replaceChildren();
  terms.forEach((term) => { const item=document.createElement("div"); item.className="hamiltonian-term"; const operator=document.createElement("span"), coefficient=document.createElement("span"); replaceWithMath(operator,M.hamiltonianTerm(term)); replaceWithMath(coefficient,M.numeric(term.coefficient,6,true)); item.append(operator,coefficient); dom.hamiltonianTerms.append(item); });
  if (app.checkpoint) { dom.phaseExamples.replaceChildren(); app.checkpoint.states.rows.slice(0,6).forEach((row) => { const item=document.createElement("article"); const state=document.createElement("div"); replaceWithMath(state,M.ket(row.bitstring)); const values=document.createElement("div"); const rotation=row.cost_phase_rotation===null?"":String.raw`,\quad(-\gamma_{${app.layer}}E_x)\bmod 2\pi=${M.numeric(row.cost_phase_rotation,4)}`; replaceWithMath(values,String.raw`E_x=${M.numeric(row.energy,2)},\quad\arg(a_x)=${M.numeric(row.phase,4)}${rotation}`); item.append(state,values); dom.phaseExamples.append(item); }); }
}

async function renderComparison() {
  if (!app.checkpoint || dom.comparisonSelect.value !== "compare") return;
  dom.comparisonPanel.classList.remove("hidden"); const algorithms=["penalty_x","global_grover"]; const payloads=[];
  for (const algorithm of algorithms) { const key=checkpointURL(algorithm); if (!app.checkpointCache.has(key)) app.checkpointCache.set(key,await fetchJSON(key)); payloads.push(app.checkpointCache.get(key)); }
  dom.comparisonContent.replaceChildren(); payloads.forEach((payload) => { const m=payload.metrics; const card=document.createElement("article"); card.className="comparison-card"; const title=document.createElement("h3"); title.textContent=payload.algorithm_label; card.append(title); const metrics=document.createElement("div"); metrics.className="comparison-metrics"; [["expected_hc",m.expected_hc],["p_feas",m.p_feas],["p_opt",m.p_opt],["p_opt_given_feasible",m.p_opt_given_feasible]].forEach(([field,value])=>{ const item=document.createElement("article"),label=document.createElement("span"),strong=document.createElement("strong"); replaceWithMath(label,M.symbolForField(field)); replaceWithMath(strong,M.numeric(value,6)); item.append(label,strong); metrics.append(item); }); card.append(metrics); const amplitude=document.createElement("p"); amplitude.className="panel-note"; appendMath(amplitude,M.stateAmplitude(payload.selected_state.real,payload.selected_state.imag)); appendText(amplitude," · "); const top=payload.states.rows.find((row)=>row.basis_index===m.top_state_index); appendMath(amplitude,top?String.raw`x_{\max}=${M.ket(top.bitstring)}`:String.raw`x_{\max}=${m.top_state_index}`); card.append(amplitude); const edges=payload.graph.edges.slice().sort((a,b)=>b.selection_probability-a.selection_probability).slice(0,6); const edgeList=document.createElement("div"); edgeList.className="edge-difference-list"; edges.forEach((edge)=>{ const item=document.createElement("span"); replaceWithMath(item,M.edgeMarginal(edge.u,edge.v,edge.selection_probability)); edgeList.append(item); }); card.append(edgeList); dom.comparisonContent.append(card); });
}

function togglePresentation() { app.presentation=!app.presentation; document.body.classList.toggle("presentation",app.presentation); dom.modeToggle.textContent=app.presentation?"Analysis Mode":"Presentation Mode"; setTimeout(renderCanvases,80); }

function notationAudit() {
  const visibleText = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode;
    const parent = node.parentElement;
    if (!parent || parent.closest("script,style,.katex-mathml,.hidden")) continue;
    const style = getComputedStyle(parent);
    if (style.display === "none" || style.visibility === "hidden" || Number(style.opacity) === 0) continue;
    if (!parent.getClientRects().length) continue;
    const value = node.nodeValue.trim();
    if (value) visibleText.push(value);
  }
  const textValue = visibleText.join(" ");
  const forbidden = [
    "p_feas", "p_opt", "p_opt_given_feasible", "expectation_energy",
    "state_real", "state_imag", "delta_probability", "delta_energy",
    "before_cost", "after_cost", "after_mixer",
  ];
  const engineeringIdentifiers = forbidden.filter((token) => new RegExp(`(^|[^A-Za-z0-9])${token}([^A-Za-z0-9]|$)`).test(textValue));
  const dynamicEngineering = textValue.match(/\b(?:gamma|beta)_\d+\b/g) || [];
  const rawDirac = textValue.match(/(?:<[^>]{0,40}\||\|[^<]{0,40}>)/g) || [];
  const rawLatex = textValue.match(/\\(?:langle|rangle|lvert|rvert|gamma|beta|psi|sum|frac|operatorname)\b/g) || [];
  const unrendered = [...document.querySelectorAll("[data-math]:not([data-math-rendered]),[data-math-key]:not([data-math-rendered])")]
    .filter((element) => getComputedStyle(element).display !== "none").length;
  const katexErrors = document.querySelectorAll(".katex-error").length;
  const formulaOverflow = [...document.querySelectorAll("[data-math-rendered]")].filter((element) => {
    if (!element.getClientRects().length || element.closest(".state-table-wrap,.parameter-table-wrap,.circuit-window,.formula-box")) return false;
    const rect = element.getBoundingClientRect();
    return rect.left < -1 || rect.right > window.innerWidth + 1;
  }).length;
  return {
    visible_text: textValue,
    engineering_identifiers: [...new Set(engineeringIdentifiers.concat(dynamicEngineering))],
    raw_dirac_notation: [...new Set(rawDirac)],
    raw_latex_commands: [...new Set(rawLatex)],
    unrendered_math: unrendered,
    katex_errors: katexErrors,
    formula_overflow: formulaOverflow,
    passed: engineeringIdentifiers.length + dynamicEngineering.length + rawDirac.length + rawLatex.length + unrendered + katexErrors + formulaOverflow === 0,
  };
}

window.__QAOA_NOTATION_AUDIT__ = notationAudit;
function fatal(error) { console.error(error); stopAutoplay(); dom.errorMessage.textContent="A required local visualization resource or frozen-artifact validation failed. See the server console for diagnostic details."; dom.errorOverlay.classList.remove("hidden"); dom.loadingOverlay.classList.add("hidden"); }

async function initialize() {
  cacheDom(); bindEvents();
  try {
    M.renderStatic();
    app.catalog = await fetchJSON("/api/catalog"); initializeControls();
    dom.validationBadge.textContent = app.catalog.validation.all_checks_passed ? "✓ frozen artifact checks passed" : "validation failed";
    dom.validationBadge.classList.toggle("valid", app.catalog.validation.all_checks_passed);
    await loadRun();
    window.__QAOA_READY__ = true;
  } catch (error) { fatal(error); }
}

initialize();

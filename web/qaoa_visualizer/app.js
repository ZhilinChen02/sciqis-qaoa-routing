"use strict";

const elements = {
  method: document.querySelector("#method-select"),
  depth: document.querySelector("#depth-select"),
  seed: document.querySelector("#seed-select"),
  load: document.querySelector("#load-run"),
  loading: document.querySelector("#loading"),
  error: document.querySelector("#error"),
  workspace: document.querySelector("#workspace"),
  runId: document.querySelector("#run-id"),
  methodTitle: document.querySelector("#method-title"),
  methodExplanation: document.querySelector("#method-explanation"),
  representation: document.querySelector("#representation"),
  circuit: document.querySelector("#circuit"),
  previous: document.querySelector("#previous"),
  play: document.querySelector("#play"),
  next: document.querySelector("#next"),
  gateByGate: document.querySelector("#gate-by-gate"),
  speed: document.querySelector("#speed-select"),
  frameSlider: document.querySelector("#frame-slider"),
  frameOutput: document.querySelector("#frame-output"),
  stageLabel: document.querySelector("#stage-label"),
  energyValue: document.querySelector("#energy-value"),
  lossValue: document.querySelector("#loss-value"),
  lossDescription: document.querySelector("#loss-description"),
  bspValue: document.querySelector("#bsp-value"),
  routeValue: document.querySelector("#route-value"),
  routeCost: document.querySelector("#route-cost"),
  energyDelta: document.querySelector("#energy-delta"),
  energyChart: document.querySelector("#energy-chart"),
  parameterChart: document.querySelector("#parameter-chart"),
  parameterValues: document.querySelector("#parameter-values"),
  distribution: document.querySelector("#distribution"),
};

const state = {
  catalog: null,
  run: null,
  frameIndex: 0,
  stageIndex: 0,
  timer: null,
};

const unique = (values) => [...new Set(values)];
const formatAngle = (value) => Number(value).toFixed(4);
const formatNumber = (value, digits = 6) =>
  value === null || value === undefined || !Number.isFinite(Number(value))
    ? "—"
    : Number(value).toFixed(digits);
const formatPercent = (value) =>
  value === null || value === undefined || !Number.isFinite(Number(value))
    ? "—"
    : `${(100 * Number(value)).toFixed(3)}%`;

function setOptions(select, items, valueFor, labelFor, preferred = null) {
  const previous = preferred ?? select.value;
  select.replaceChildren();
  items.forEach((item) => {
    const option = document.createElement("option");
    option.value = String(valueFor(item));
    option.textContent = String(labelFor(item));
    select.append(option);
  });
  const available = [...select.options].some((option) => option.value === String(previous));
  if (available) select.value = String(previous);
}

function methodRuns() {
  return state.catalog.runs.filter((run) => run.method === elements.method.value);
}

function depthRuns() {
  return methodRuns().filter((run) => run.depth === Number(elements.depth.value));
}

function updateDepths(preferred = null) {
  const depths = unique(methodRuns().map((run) => run.depth)).sort((a, b) => a - b);
  setOptions(elements.depth, depths, (item) => item, (item) => `p = ${item}`, preferred);
  updateSeeds();
}

function updateSeeds(preferred = null) {
  const seeds = unique(depthRuns().map((run) => run.seed)).sort((a, b) => a - b);
  setOptions(elements.seed, seeds, (item) => item, (item) => item, preferred);
}

function selectedCatalogRun() {
  return state.catalog.runs.find(
    (run) =>
      run.method === elements.method.value &&
      run.depth === Number(elements.depth.value) &&
      run.seed === Number(elements.seed.value),
  );
}

async function fetchJSON(url) {
  const response = await fetch(url, { cache: "no-store" });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || payload.error || `HTTP ${response.status}`);
  return payload;
}

async function initialize() {
  try {
    state.catalog = await fetchJSON("/api/runs");
    const methods = unique(state.catalog.runs.map((run) => run.method)).map((method) =>
      state.catalog.runs.find((run) => run.method === method),
    );
    setOptions(
      elements.method,
      methods,
      (run) => run.method,
      (run) => run.method_short_label,
      "gm_th_qaoa",
    );
    updateDepths("3");
    updateSeeds("2601");
    await loadSelectedRun();
  } catch (error) {
    showError(error);
  }
}

async function loadSelectedRun() {
  const selected = selectedCatalogRun();
  if (!selected) return;
  stopPlayback();
  elements.loading.hidden = false;
  elements.loading.textContent = "Reconstructing each layer's quantum state and energy…";
  elements.error.hidden = true;
  elements.workspace.hidden = true;
  try {
    state.run = await fetchJSON(`/api/runs/${encodeURIComponent(selected.run_id)}`);
    state.frameIndex = 0;
    state.stageIndex = lastStageIndex();
    elements.frameSlider.max = String(state.run.frames.length - 1);
    elements.frameSlider.value = "0";
    buildCircuit();
    render();
    elements.workspace.hidden = false;
  } catch (error) {
    showError(error);
  } finally {
    elements.loading.hidden = true;
  }
}

function showError(error) {
  elements.loading.hidden = true;
  elements.workspace.hidden = true;
  elements.error.hidden = false;
  elements.error.textContent = `Unable to load the visualization: ${error.message || error}`;
}

function currentFrame() {
  return state.run.frames[state.frameIndex];
}

function lastStageIndex() {
  if (!state.run) return 0;
  const frame = currentFrame();
  return Math.max(0, frame.stages.length - 1);
}

function currentStage() {
  const frame = currentFrame();
  if (!frame.stages.length) return null;
  state.stageIndex = Math.min(state.stageIndex, frame.stages.length - 1);
  return frame.stages[state.stageIndex];
}

function buildCircuit() {
  const circuit = state.run.circuit;
  elements.circuit.replaceChildren();
  elements.representation.textContent = circuit.representation;

  const initial = document.createElement("div");
  initial.className = "wire-label";
  initial.innerHTML = `<b>${circuit.initial.symbol}</b><small>${circuit.initial.label}</small>`;
  elements.circuit.append(initial);

  for (let layer = 1; layer <= state.run.depth; layer += 1) {
    const group = document.createElement("div");
    group.className = "layer-group";
    const number = document.createElement("span");
    number.className = "layer-number";
    number.textContent = `LAYER ${layer}`;
    group.append(number);
    group.append(makeGate("phase", layer, circuit.phase));
    group.append(makeGate("mixer", layer, circuit.mixer));
    elements.circuit.append(group);
  }

  const measurement = document.createElement("div");
  measurement.className = "measure";
  measurement.innerHTML = `<b>${circuit.measurement.symbol}</b><small>${circuit.measurement.label}</small>`;
  elements.circuit.append(measurement);
}

function makeGate(kind, layer, description) {
  const gate = document.createElement("div");
  gate.className = `gate ${kind}`;
  gate.dataset.kind = kind;
  gate.dataset.layer = String(layer);
  const parameter = kind === "phase" ? `γ${layer}` : `β${layer}`;
  gate.innerHTML = `
    <span class="gate-symbol">${description.symbol}</span>
    <span class="gate-angle" data-parameter="${parameter}">${parameter} = —</span>
    <small>${description.label}</small>
  `;
  gate.title = description.formula;
  return gate;
}

function render() {
  if (!state.run) return;
  const frame = currentFrame();
  const stage = currentStage();
  elements.runId.textContent = `${state.run.run_id} · ${frame.in_bounds ? "IN BOUNDS" : "OUT OF BOUNDS"}`;
  elements.methodTitle.textContent = state.run.method_label;
  elements.methodExplanation.textContent = state.run.method_explanation;
  elements.frameSlider.value = String(state.frameIndex);
  elements.frameOutput.textContent = `${frame.evaluation} / ${state.run.frames.length}`;

  renderCircuit(frame, stage);
  renderMetrics(frame, stage);
  renderParameterValues(frame);
  drawEnergyChart(stage);
  drawParameterChart();
  renderDistribution(stage);
}

function renderCircuit(frame, stage) {
  elements.circuit.querySelectorAll(".gate").forEach((gate) => {
    const layer = Number(gate.dataset.layer);
    const kind = gate.dataset.kind;
    const value = kind === "phase" ? frame.gammas[layer - 1] : frame.betas[layer - 1];
    const symbol = kind === "phase" ? "γ" : "β";
    gate.querySelector(".gate-angle").textContent = `${symbol}${layer} = ${formatAngle(value)}`;
    const active = stage && stage.kind === kind && stage.layer === layer;
    gate.classList.toggle("active", Boolean(active));
  });

  if (!frame.in_bounds) {
    elements.stageLabel.textContent = "Out-of-bounds parameters: the optimizer recorded this request, but no statevector simulation was run.";
  } else if (!stage || stage.kind === "initial") {
    elements.stageLabel.textContent = "Initial feasible state";
  } else {
    const symbol = stage.kind === "phase" ? "γ" : "β";
    const operation = stage.kind === "phase" ? state.run.circuit.phase.label : state.run.circuit.mixer.label;
    elements.stageLabel.textContent = `Layer ${stage.layer} · ${operation} · ${symbol}${stage.layer} = ${formatAngle(stage.angle)}`;
  }
}

function renderMetrics(frame, stage) {
  elements.energyValue.textContent = formatNumber(stage?.expected_cost ?? frame.expected_cost, 6);
  elements.lossValue.textContent = formatNumber(frame.loss, 6);
  elements.lossDescription.textContent =
    state.run.method === "gm_qaoa_expectation" ? "normalized ⟨H_C⟩" : "−P(C < incumbent)";
  elements.bspValue.textContent = formatPercent(stage?.bsp ?? frame.bsp);

  if (!stage) {
    elements.routeValue.textContent = "—";
    elements.routeCost.textContent = "No statevector evolution was run";
    return;
  }
  const route = state.run.routes[stage.most_probable_route_id];
  elements.routeValue.textContent = route.node_sequence.join(" → ");
  elements.routeCost.textContent = `cost ${route.routing_cost} · P ${formatPercent(stage.probabilities[route.route_id])}`;
}

function renderParameterValues(frame) {
  elements.parameterValues.replaceChildren();
  frame.gammas.forEach((value, index) => {
    const chip = document.createElement("span");
    chip.className = "parameter-chip gamma";
    chip.textContent = `γ${index + 1} ${formatAngle(value)}`;
    elements.parameterValues.append(chip);
  });
  frame.betas.forEach((value, index) => {
    const chip = document.createElement("span");
    chip.className = "parameter-chip beta";
    chip.textContent = `β${index + 1} ${formatAngle(value)}`;
    elements.parameterValues.append(chip);
  });
}

function svgLinePath(values, xFor, yFor) {
  let path = "";
  let drawing = false;
  values.forEach((value, index) => {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) {
      drawing = false;
      return;
    }
    path += `${drawing ? " L" : " M"} ${xFor(index).toFixed(2)} ${yFor(Number(value)).toFixed(2)}`;
    drawing = true;
  });
  return path.trim();
}

function drawEnergyChart(stage) {
  const width = 720;
  const height = 245;
  const margin = { left: 48, right: 16, top: 12, bottom: 30 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const minimum = state.run.energy.minimum_basis_cost;
  const maximum = state.run.energy.maximum_basis_cost;
  const values = state.run.frames.map((frame) => frame.expected_cost);
  const xFor = (index) => margin.left + (innerWidth * index) / Math.max(1, values.length - 1);
  const yFor = (value) => margin.top + innerHeight * (1 - (value - minimum) / (maximum - minimum));
  const ticks = [minimum, minimum + 1, minimum + 2, minimum + 3, maximum];
  const grid = ticks
    .map((tick) => {
      const y = yFor(tick);
      return `<line class="grid-line" x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}"/><text class="axis-label" x="${margin.left - 9}" y="${y + 3}" text-anchor="end">${tick.toFixed(0)}</text>`;
    })
    .join("");
  const cursorX = xFor(state.frameIndex);
  const pointValue = stage?.expected_cost ?? currentFrame().expected_cost;
  const hasPoint =
    pointValue !== null && pointValue !== undefined && Number.isFinite(Number(pointValue));
  const point = hasPoint
    ? `<circle class="chart-point" cx="${cursorX}" cy="${yFor(Number(pointValue))}" r="4"/>`
    : "";
  elements.energyChart.setAttribute("viewBox", `0 0 ${width} ${height}`);
  elements.energyChart.innerHTML = `
    ${grid}
    <line class="axis-line" x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}"/>
    <path class="series-energy" d="${svgLinePath(values, xFor, yFor)}"/>
    <line class="chart-cursor" x1="${cursorX}" y1="${margin.top}" x2="${cursorX}" y2="${height - margin.bottom}"/>
    ${point}
    <text class="axis-label" x="${margin.left}" y="${height - 8}">evaluation 1</text>
    <text class="axis-label" x="${width - margin.right}" y="${height - 8}" text-anchor="end">evaluation ${values.length}</text>
  `;

  const first = values.find((value) => value !== null && value !== undefined);
  const current = pointValue;
  if (
    first !== null &&
    first !== undefined &&
    current !== null &&
    current !== undefined &&
    Number.isFinite(Number(first)) &&
    Number.isFinite(Number(current))
  ) {
    const delta = Number(current) - Number(first);
    elements.energyDelta.textContent = `Δ ${delta >= 0 ? "+" : ""}${delta.toFixed(4)}`;
  } else {
    elements.energyDelta.textContent = "";
  }
}

function drawParameterChart() {
  const width = 720;
  const height = 180;
  const margin = { left: 48, right: 16, top: 8, bottom: 28 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const count = state.run.frames.length;
  const allValues = state.run.frames.flatMap((frame) => [...frame.gammas, ...frame.betas]);
  const minimum = Math.min(0, ...allValues);
  const maximum = Math.max(2 * Math.PI, ...allValues);
  const xFor = (index) => margin.left + (innerWidth * index) / Math.max(1, count - 1);
  const yFor = (value) => margin.top + innerHeight * (1 - (value - minimum) / (maximum - minimum));
  const ticks = [minimum, Math.PI, maximum];
  const grid = ticks
    .map((tick) => {
      const y = yFor(tick);
      const label = Math.abs(tick - Math.PI) < 1e-9 ? "π" : tick.toFixed(2);
      return `<line class="grid-line" x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}"/><text class="axis-label" x="${margin.left - 9}" y="${y + 3}" text-anchor="end">${label}</text>`;
    })
    .join("");
  const series = [];
  for (let layer = 0; layer < state.run.depth; layer += 1) {
    const gammas = state.run.frames.map((frame) => frame.gammas[layer]);
    const betas = state.run.frames.map((frame) => frame.betas[layer]);
    series.push(`<path class="series-gamma" d="${svgLinePath(gammas, xFor, yFor)}"/>`);
    series.push(`<path class="series-beta" d="${svgLinePath(betas, xFor, yFor)}"/>`);
  }
  const cursorX = xFor(state.frameIndex);
  elements.parameterChart.setAttribute("viewBox", `0 0 ${width} ${height}`);
  elements.parameterChart.innerHTML = `
    ${grid}
    <line class="axis-line" x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}"/>
    ${series.join("")}
    <line class="chart-cursor" x1="${cursorX}" y1="${margin.top}" x2="${cursorX}" y2="${height - margin.bottom}"/>
    <text class="axis-label" x="${margin.left}" y="${height - 7}">1</text>
    <text class="axis-label" x="${width - margin.right}" y="${height - 7}" text-anchor="end">${count}</text>
  `;
}

function renderDistribution(stage) {
  elements.distribution.replaceChildren();
  const probabilities = stage?.probabilities ?? state.run.routes.map(() => 0);
  const maximum = Math.max(0.000001, ...probabilities);
  state.run.routes.forEach((route) => {
    const probability = probabilities[route.route_id] ?? 0;
    const wrap = document.createElement("div");
    wrap.className = "route-bar-wrap";
    wrap.title = `${route.node_sequence.join(" → ")}\ncost ${route.routing_cost}\nP = ${formatPercent(probability)}`;
    const area = document.createElement("div");
    area.className = "route-bar-area";
    const bar = document.createElement("div");
    bar.className = `route-bar${route.exact_optimal ? " optimal" : ""}`;
    bar.style.height = probability <= 0 ? "0" : `${Math.max(1, (100 * probability) / maximum)}%`;
    area.append(bar);
    const label = document.createElement("b");
    label.textContent = `P${route.route_id}`;
    const cost = document.createElement("small");
    cost.textContent = `C${route.routing_cost}`;
    wrap.append(area, label, cost);
    elements.distribution.append(wrap);
  });
}

function stopPlayback() {
  if (state.timer !== null) window.clearInterval(state.timer);
  state.timer = null;
  elements.play.textContent = "▶ Play";
}

function togglePlayback() {
  if (state.timer !== null) {
    stopPlayback();
    return;
  }
  if (state.frameIndex >= state.run.frames.length - 1 && state.stageIndex >= lastStageIndex()) {
    state.frameIndex = 0;
    state.stageIndex = elements.gateByGate.checked ? 0 : lastStageIndex();
  }
  elements.play.textContent = "Ⅱ Pause";
  state.timer = window.setInterval(advance, Number(elements.speed.value));
}

function advance() {
  const frame = currentFrame();
  if (elements.gateByGate.checked && frame.stages.length && state.stageIndex < frame.stages.length - 1) {
    state.stageIndex += 1;
    render();
    return;
  }
  if (state.frameIndex >= state.run.frames.length - 1) {
    stopPlayback();
    return;
  }
  state.frameIndex += 1;
  state.stageIndex = elements.gateByGate.checked ? 0 : lastStageIndex();
  render();
}

function retreat() {
  stopPlayback();
  const frame = currentFrame();
  if (elements.gateByGate.checked && frame.stages.length && state.stageIndex > 0) {
    state.stageIndex -= 1;
  } else if (state.frameIndex > 0) {
    state.frameIndex -= 1;
    state.stageIndex = elements.gateByGate.checked ? lastStageIndex() : lastStageIndex();
  }
  render();
}

function moveNext() {
  stopPlayback();
  advance();
}

elements.method.addEventListener("change", () => updateDepths());
elements.depth.addEventListener("change", () => updateSeeds());
elements.load.addEventListener("click", loadSelectedRun);
elements.play.addEventListener("click", togglePlayback);
elements.previous.addEventListener("click", retreat);
elements.next.addEventListener("click", moveNext);
elements.speed.addEventListener("change", () => {
  if (state.timer !== null) {
    stopPlayback();
    togglePlayback();
  }
});
elements.gateByGate.addEventListener("change", () => {
  stopPlayback();
  state.stageIndex = elements.gateByGate.checked ? 0 : lastStageIndex();
  render();
});
elements.frameSlider.addEventListener("input", () => {
  stopPlayback();
  state.frameIndex = Number(elements.frameSlider.value);
  state.stageIndex = elements.gateByGate.checked ? 0 : lastStageIndex();
  render();
});

initialize();

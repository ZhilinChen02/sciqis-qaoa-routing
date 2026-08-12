"use strict";

const COLORS = {
  text: "#eef8f7",
  muted: "#96adb2",
  grid: "rgba(164,211,220,0.13)",
  teal: "#3ee0c1",
  cyan: "#57c7ff",
  gold: "#ffc857",
  orange: "#ff8a5b",
  red: "#ff6680",
  purple: "#a98bff",
  green: "#7be495",
  dark: "#071015",
};

const appState = {
  catalog: null,
  run: null,
  cache: new Map(),
  checkpoint: 0,
  selectedDisplayPosition: null,
  selectedExternalState: null,
  amplitudeMode: "magnitude",
  landscapeMetric: "expected_hc",
  autoplayTimer: null,
  presentation: false,
  sceneIndex: 0,
  phaseHitRegions: [],
  complexHitRegions: [],
  landscapeHitRegions: [],
};

const dom = {};
const SVG_NS = "http://www.w3.org/2000/svg";

function byId(id) {
  return document.getElementById(id);
}

function cacheDom() {
  [
    "algorithmSelect", "depthSelect", "checkpointSlider", "checkpointOutput",
    "previousButton", "playButton", "nextButton", "speedSelect",
    "validationBadge", "loadingOverlay", "errorOverlay", "errorMessage",
    "runSource", "circuitTimeline", "checkpointTeaching", "energyCanvas",
    "energyLegend", "energyDelta", "pFeasValue", "pOptValue", "invalidValue",
    "metricsCanvas", "structuralBadge", "metricInterpretation", "probabilityFlow",
    "probabilityScope", "complexCanvas", "complexNote", "phaseCanvas",
    "phaseSampling", "landscapeCanvas", "landscapeCounts", "energyInspector",
    "geometrySvg", "geometryAnimate", "routeSvg", "routeDetails",
    "selectedStateBadge", "optimizerCanvas", "landscapeMetric", "optimizerDetails",
    "tooltip", "environmentNote", "presentationToggle", "presentationBar",
    "presentationExit", "scenePrevious", "sceneNext", "sceneCounter", "sceneTitle",
    "sceneMessage", "sceneComparison",
  ].forEach((id) => { dom[id] = byId(id); });
}

async function fetchJSON(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${url} returned HTTP ${response.status}`);
  return response.json();
}

function formatProbability(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "unavailable";
  if (value === 0) return "0";
  if (value < 1e-4) return value.toExponential(3);
  return value.toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
}

function formatNumber(value, digits = 5) {
  if (value === null || value === undefined || Number.isNaN(value)) return "unavailable";
  return Number(value).toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
}

function frame() {
  return appState.run.frames[appState.checkpoint];
}

function stateValue(position) {
  return frame().state_values[position];
}

function stateAt(position) {
  return appState.run.display_states[position];
}

function setupCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(260, Math.round(rect.width));
  const height = Math.max(180, Math.round(rect.height));
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  return { context, width, height };
}

function drawText(context, text, x, y, options = {}) {
  context.save();
  context.fillStyle = options.color || COLORS.muted;
  context.font = options.font || "12px Inter, system-ui, sans-serif";
  context.textAlign = options.align || "left";
  context.textBaseline = options.baseline || "alphabetic";
  context.fillText(String(text), x, y);
  context.restore();
}

function drawPlotFrame(context, width, height, options = {}) {
  const margins = Object.assign({ left: 58, right: 22, top: 24, bottom: 42 }, options.margins || {});
  const plot = {
    left: margins.left,
    top: margins.top,
    right: width - margins.right,
    bottom: height - margins.bottom,
  };
  context.save();
  context.strokeStyle = COLORS.grid;
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(plot.left, plot.top);
  context.lineTo(plot.left, plot.bottom);
  context.lineTo(plot.right, plot.bottom);
  context.stroke();
  context.restore();
  return plot;
}

function linePath(context, values, xFor, yFor, color, selected, dashed = false) {
  context.save();
  context.strokeStyle = color;
  context.lineWidth = 2.3;
  if (dashed) context.setLineDash([5, 5]);
  context.beginPath();
  values.forEach((value, index) => {
    const x = xFor(index);
    const y = yFor(value);
    if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
  });
  context.stroke();
  values.forEach((value, index) => {
    context.beginPath();
    context.arc(xFor(index), yFor(value), index === selected ? 5.8 : 3.2, 0, Math.PI * 2);
    context.fillStyle = index === selected ? COLORS.text : color;
    context.fill();
    if (index === selected) {
      context.strokeStyle = color;
      context.lineWidth = 2;
      context.stroke();
    }
  });
  context.restore();
}

function initializeSelectors() {
  appState.catalog.algorithms.forEach((algorithm) => {
    const option = document.createElement("option");
    option.value = algorithm.id;
    option.textContent = algorithm.label;
    dom.algorithmSelect.append(option);
  });
  dom.algorithmSelect.value = appState.catalog.default.algorithm;
  dom.depthSelect.value = String(appState.catalog.default.depth);
}

function bindEvents() {
  dom.algorithmSelect.addEventListener("change", () => loadSelectedRun(0));
  dom.depthSelect.addEventListener("change", () => loadSelectedRun(0));
  dom.checkpointSlider.addEventListener("input", () => setCheckpoint(Number(dom.checkpointSlider.value)));
  dom.previousButton.addEventListener("click", () => setCheckpoint(appState.checkpoint - 1));
  dom.nextButton.addEventListener("click", () => setCheckpoint(appState.checkpoint + 1));
  dom.playButton.addEventListener("click", toggleAutoplay);
  dom.speedSelect.addEventListener("change", () => { if (appState.autoplayTimer) { stopAutoplay(); startAutoplay(); } });
  document.querySelectorAll("[data-amplitude-mode]").forEach((button) => {
    button.addEventListener("click", () => {
      appState.amplitudeMode = button.dataset.amplitudeMode;
      document.querySelectorAll("[data-amplitude-mode]").forEach((item) => item.classList.toggle("active", item === button));
      drawComplexAmplitudes();
    });
  });
  dom.landscapeMetric.addEventListener("change", () => {
    appState.landscapeMetric = dom.landscapeMetric.value;
    drawOptimizerLandscape();
  });
  dom.geometryAnimate.addEventListener("click", animateGeometry);
  dom.presentationToggle.addEventListener("click", enterPresentation);
  dom.presentationExit.addEventListener("click", exitPresentation);
  dom.scenePrevious.addEventListener("click", () => showScene(appState.sceneIndex - 1));
  dom.sceneNext.addEventListener("click", () => showScene(appState.sceneIndex + 1));
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && appState.presentation) exitPresentation();
    if (appState.presentation && event.key === "ArrowRight") showScene(appState.sceneIndex + 1);
    if (appState.presentation && event.key === "ArrowLeft") showScene(appState.sceneIndex - 1);
    if (!appState.presentation && event.key === " ") { event.preventDefault(); toggleAutoplay(); }
  });
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (appState.run) renderAll(); }, 120);
  });
  attachCanvasInteraction(dom.phaseCanvas, () => appState.phaseHitRegions);
  attachCanvasInteraction(dom.complexCanvas, () => appState.complexHitRegions);
}

function attachCanvasInteraction(canvas, regions) {
  canvas.addEventListener("mousemove", (event) => {
    const hit = nearestHit(canvas, event, regions(), 12);
    if (!hit) return hideTooltip();
    const state = stateAt(hit.position);
    const value = stateValue(hit.position);
    showTooltip(event, stateTooltip(state, value));
  });
  canvas.addEventListener("mouseleave", hideTooltip);
  canvas.addEventListener("click", (event) => {
    const hit = nearestHit(canvas, event, regions(), 14);
    if (hit) selectDisplayState(hit.position);
  });
}

function nearestHit(canvas, event, regions, threshold) {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  let result = null;
  let best = threshold * threshold;
  regions.forEach((region) => {
    const distance = (region.x - x) ** 2 + (region.y - y) ** 2;
    if (distance <= best) { best = distance; result = region; }
  });
  return result;
}

function stateTooltip(state, value) {
  const validity = state.optimal ? "exact optimum" : (state.feasible ? "feasible" : "infeasible");
  return `<strong>${state.identity}</strong><br>${validity} · E=${formatNumber(state.energy, 3)}<br>p=${formatProbability(value[3])} · φ=${formatNumber(value[4], 3)} rad`;
}

function showTooltip(event, html) {
  dom.tooltip.innerHTML = html;
  dom.tooltip.classList.remove("hidden");
  const x = Math.min(window.innerWidth - 350, event.clientX + 14);
  const y = Math.min(window.innerHeight - 100, event.clientY + 14);
  dom.tooltip.style.left = `${Math.max(8, x)}px`;
  dom.tooltip.style.top = `${Math.max(8, y)}px`;
}

function hideTooltip() { dom.tooltip.classList.add("hidden"); }

async function loadSelectedRun(checkpoint = 0) {
  stopAutoplay();
  const algorithm = dom.algorithmSelect.value;
  const depth = Number(dom.depthSelect.value);
  await loadRun(algorithm, depth, checkpoint);
}

async function loadRun(algorithm, depth, checkpoint = 0) {
  const key = `${algorithm}:p${depth}`;
  dom.loadingOverlay.classList.remove("hidden");
  try {
    if (!appState.cache.has(key)) {
      appState.cache.set(key, await fetchJSON(`/api/run/${algorithm}/${depth}`));
    }
    appState.run = appState.cache.get(key);
    dom.algorithmSelect.value = algorithm;
    dom.depthSelect.value = String(depth);
    appState.selectedDisplayPosition = null;
    appState.selectedExternalState = null;
    dom.checkpointSlider.max = String(appState.run.frames.length - 1);
    setCheckpoint(Math.max(0, Math.min(checkpoint, appState.run.frames.length - 1)), false);
  } catch (error) {
    showFatal(error);
  } finally {
    dom.loadingOverlay.classList.add("hidden");
  }
}

function setCheckpoint(index, doRender = true) {
  if (!appState.run) return;
  appState.checkpoint = Math.max(0, Math.min(index, appState.run.frames.length - 1));
  dom.checkpointSlider.value = String(appState.checkpoint);
  dom.checkpointOutput.textContent = frame().checkpoint;
  dom.previousButton.disabled = appState.checkpoint === 0;
  dom.nextButton.disabled = appState.checkpoint === appState.run.frames.length - 1;
  if (doRender) renderAll(); else renderAll();
}

function startAutoplay() {
  if (!appState.run || appState.autoplayTimer) return;
  if (appState.checkpoint >= appState.run.frames.length - 1) setCheckpoint(0);
  dom.playButton.textContent = "Ⅱ Pause";
  appState.autoplayTimer = window.setInterval(() => {
    if (appState.checkpoint >= appState.run.frames.length - 1) return stopAutoplay();
    setCheckpoint(appState.checkpoint + 1);
  }, Number(dom.speedSelect.value));
}

function stopAutoplay() {
  if (appState.autoplayTimer) window.clearInterval(appState.autoplayTimer);
  appState.autoplayTimer = null;
  if (dom.playButton) dom.playButton.textContent = "▶ Play";
}

function toggleAutoplay() { appState.autoplayTimer ? stopAutoplay() : startAutoplay(); }

function renderAll() {
  if (!appState.run) return;
  renderCircuit();
  drawEnergy();
  drawMetrics();
  renderProbabilityFlow();
  drawComplexAmplitudes();
  drawPhaseScatter();
  drawEnergyLandscape();
  renderGeometry();
  renderRoute();
  drawOptimizerLandscape();
}

function renderCircuit() {
  dom.circuitTimeline.replaceChildren();
  appState.run.circuit.gates.forEach((gate) => {
    const element = document.createElement("article");
    element.className = `circuit-gate ${gate.kind}`;
    if (gate.checkpoint_index !== null && gate.checkpoint_index < appState.checkpoint) element.classList.add("past");
    if (gate.checkpoint_index === appState.checkpoint) element.classList.add("active");
    if (gate.kind === "measurement" && appState.checkpoint === appState.run.frames.length - 1) element.classList.add("past");
    const symbol = document.createElement("strong");
    symbol.className = "gate-symbol";
    symbol.textContent = gate.label;
    const angle = document.createElement("span");
    angle.className = "gate-angle";
    angle.textContent = gate.angle === undefined ? "" : `${gate.angle_name} = ${formatNumber(gate.angle, 5)}`;
    const detail = document.createElement("span");
    detail.className = "gate-detail";
    detail.textContent = gate.detail;
    element.append(symbol, angle, detail);
    dom.circuitTimeline.append(element);
  });
  dom.checkpointTeaching.textContent = `${frame().checkpoint}: ${frame().teaching}`;
  dom.runSource.textContent = `${appState.run.algorithm_short_label} · p=${appState.run.depth} · ${appState.run.search_dimension.toLocaleString()} basis states · saved seed ${appState.run.seed}`;
}

function drawEnergy() {
  const { context, width, height } = setupCanvas(dom.energyCanvas);
  const frames = appState.run.frames;
  const total = frames.map((item) => item.metrics.expected_hc);
  const route = frames.map((item) => item.metrics.expected_routing_term);
  const penalty = frames.map((item) => item.metrics.expected_penalty_contribution);
  const series = [{ label: "⟨H_C⟩ / total", values: total, color: COLORS.teal }];
  if (!appState.run.structural_feasibility && route.every((value) => value !== null) && penalty.every((value) => value !== null)) {
    series.push({ label: "⟨H_route⟩", values: route, color: COLORS.cyan });
    series.push({ label: "A⟨H_penalty⟩", values: penalty, color: COLORS.orange, dashed: true });
  }
  const all = series.flatMap((item) => item.values);
  let yMin = Math.min(...all);
  let yMax = Math.max(...all);
  const pad = Math.max(1, (yMax - yMin) * 0.16);
  yMin = Math.max(0, yMin - pad);
  yMax += pad;
  const plot = drawPlotFrame(context, width, height);
  const xFor = (index) => plot.left + (frames.length === 1 ? 0 : index * (plot.right - plot.left) / (frames.length - 1));
  const yFor = (value) => plot.bottom - (value - yMin) * (plot.bottom - plot.top) / Math.max(1e-12, yMax - yMin);
  for (let tick = 0; tick <= 4; tick += 1) {
    const value = yMin + tick * (yMax - yMin) / 4;
    const y = yFor(value);
    context.strokeStyle = COLORS.grid;
    context.beginPath(); context.moveTo(plot.left, y); context.lineTo(plot.right, y); context.stroke();
    drawText(context, formatNumber(value, 1), plot.left - 8, y, { align: "right", baseline: "middle" });
  }
  series.forEach((item) => linePath(context, item.values, xFor, yFor, item.color, appState.checkpoint, item.dashed));
  frames.forEach((item, index) => drawText(context, item.checkpoint, xFor(index), plot.bottom + 19, { align: "center", font: "11px Inter, sans-serif" }));
  drawText(context, "expected energy", 10, plot.top - 5, { color: COLORS.text, font: "11px Inter, sans-serif" });
  dom.energyLegend.replaceChildren();
  series.forEach((item) => {
    const span = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.style.background = item.color;
    span.append(swatch, document.createTextNode(item.label));
    dom.energyLegend.append(span);
  });
  if (appState.checkpoint > 0) {
    const previous = frames[appState.checkpoint - 1].metrics.expected_hc;
    const delta = total[appState.checkpoint] - previous;
    dom.energyDelta.textContent = frame().operation === "cost"
      ? `cost-step Δ = ${delta.toExponential(2)} ≈ 0`
      : `mixer-step Δ = ${formatNumber(delta, 5)}`;
  } else dom.energyDelta.textContent = "initial expectation";
}

function drawMetrics() {
  const current = frame().metrics;
  dom.pFeasValue.textContent = formatProbability(current.p_feas);
  dom.pOptValue.textContent = formatProbability(current.p_opt);
  dom.invalidValue.textContent = formatProbability(current.invalid_mass);
  dom.structuralBadge.classList.toggle("hidden", !appState.run.structural_feasibility);
  dom.metricInterpretation.textContent = appState.run.structural_feasibility
    ? "Every logical basis state is a valid route, so p_feas≈1 is structural rather than an optimized achievement."
    : "Feasible and optimal values are exact statevector probability masses, not sample frequencies.";
  const { context, width, height } = setupCanvas(dom.metricsCanvas);
  const frames = appState.run.frames;
  const plot = drawPlotFrame(context, width, height, { margins: { left: 44, right: 18, top: 18, bottom: 38 } });
  const xFor = (index) => plot.left + index * (plot.right - plot.left) / Math.max(1, frames.length - 1);
  const yFor = (value) => plot.bottom - value * (plot.bottom - plot.top);
  [0, 0.5, 1].forEach((value) => {
    const y = yFor(value);
    context.strokeStyle = COLORS.grid;
    context.beginPath(); context.moveTo(plot.left, y); context.lineTo(plot.right, y); context.stroke();
    drawText(context, value.toFixed(1), plot.left - 7, y, { align: "right", baseline: "middle" });
  });
  linePath(context, frames.map((item) => item.metrics.p_feas), xFor, yFor, COLORS.green, appState.checkpoint);
  linePath(context, frames.map((item) => item.metrics.p_opt), xFor, yFor, COLORS.gold, appState.checkpoint);
  linePath(context, frames.map((item) => item.metrics.invalid_mass), xFor, yFor, COLORS.red, appState.checkpoint, true);
  frames.forEach((item, index) => drawText(context, item.checkpoint.replace("Mixer", "Mix"), xFor(index), plot.bottom + 17, { align: "center", font: "10px Inter, sans-serif" }));
  drawText(context, "green p_feas · gold p_opt · red invalid", plot.left, 11, { font: "10px Inter, sans-serif" });
}

function renderProbabilityFlow() {
  const items = frame().probability_flow;
  const maxProbability = Math.max(...items.map((item) => item.probability), 1e-15);
  dom.probabilityFlow.replaceChildren();
  items.forEach((item) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = `probability-row ${item.category}${item.kind === "aggregate" ? " aggregate" : ""}`;
    let identity = item.identity;
    let selected = false;
    if (item.kind === "state") {
      const state = stateAt(item.display_position);
      identity = state.optimal ? `★ ${state.identity}` : state.identity;
      selected = appState.selectedDisplayPosition === item.display_position;
      row.title = `${state.validity_explanation} Energy ${state.energy}; probability ${item.probability}`;
      row.addEventListener("click", () => selectDisplayState(item.display_position));
    } else {
      identity = `${item.identity} (${item.state_count.toLocaleString()} states)`;
      row.disabled = true;
    }
    row.classList.toggle("selected", selected);
    const name = document.createElement("span");
    name.className = "probability-identity";
    name.textContent = identity;
    const track = document.createElement("span");
    track.className = "probability-track";
    const fill = document.createElement("span");
    fill.className = "probability-fill";
    fill.style.width = `${100 * item.probability / maxProbability}%`;
    track.append(fill);
    const value = document.createElement("span");
    value.className = "probability-value";
    value.textContent = formatProbability(item.probability);
    row.append(name, track, value);
    dom.probabilityFlow.append(row);
  });
  dom.probabilityScope.textContent = appState.run.structural_feasibility
    ? "all 20 logical feasible routes"
    : "optimum + top 7 feasible + top 7 infeasible + exact aggregate remainder";
}

function categoryColor(state) {
  return state.optimal ? COLORS.gold : (state.feasible ? COLORS.green : COLORS.red);
}

function drawComplexAmplitudes() {
  const { context, width, height } = setupCanvas(dom.complexCanvas);
  const cx = width / 2;
  const cy = height / 2;
  const radius = Math.min(width, height) * 0.39;
  context.strokeStyle = COLORS.grid;
  context.lineWidth = 1;
  context.beginPath(); context.arc(cx, cy, radius, 0, Math.PI * 2); context.stroke();
  context.beginPath(); context.moveTo(cx - radius - 15, cy); context.lineTo(cx + radius + 15, cy); context.moveTo(cx, cy - radius - 15); context.lineTo(cx, cy + radius + 15); context.stroke();
  drawText(context, "Re(a)", cx + radius + 4, cy - 7, { font: "11px Inter, sans-serif" });
  drawText(context, "Im(a)", cx + 7, cy - radius - 5, { font: "11px Inter, sans-serif" });
  const positions = appState.run.complex_amplitudes.display_positions;
  const values = positions.map((position) => stateValue(position));
  const maxMagnitude = Math.max(...values.map((value) => value[2]), 1e-15);
  const maxProbability = Math.max(...values.map((value) => value[3]), 1e-15);
  appState.complexHitRegions = [];
  positions.forEach((position, index) => {
    const state = stateAt(position);
    const value = values[index];
    let scaled;
    if (appState.amplitudeMode === "phase") scaled = 0.84;
    else if (appState.amplitudeMode === "probability") scaled = 0.12 + 0.84 * value[3] / maxProbability;
    else scaled = 0.12 + 0.84 * value[2] / maxMagnitude;
    const endX = cx + radius * scaled * Math.cos(value[4]);
    const endY = cy - radius * scaled * Math.sin(value[4]);
    context.save();
    context.strokeStyle = categoryColor(state);
    context.fillStyle = categoryColor(state);
    context.globalAlpha = appState.selectedDisplayPosition === position ? 1 : 0.68;
    context.lineWidth = appState.selectedDisplayPosition === position ? 3.5 : 1.6;
    context.beginPath(); context.moveTo(cx, cy); context.lineTo(endX, endY); context.stroke();
    const angle = Math.atan2(endY - cy, endX - cx);
    context.beginPath();
    context.moveTo(endX, endY);
    context.lineTo(endX - 7 * Math.cos(angle - 0.48), endY - 7 * Math.sin(angle - 0.48));
    context.lineTo(endX - 7 * Math.cos(angle + 0.48), endY - 7 * Math.sin(angle + 0.48));
    context.closePath(); context.fill();
    context.restore();
    appState.complexHitRegions.push({ x: endX, y: endY, position });
  });
  drawText(context, `${positions.length} representative amplitudes`, 12, 18, { color: COLORS.text, font: "11px Inter, sans-serif" });
  dom.complexNote.textContent = frame().operation === "cost"
    ? "Cost checkpoint: vector angles changed from the prior frame while every vector length—and therefore every basis probability—remained invariant."
    : (frame().operation === "mixer"
      ? "Mixer checkpoint: interference changed vector lengths, redistributing probability mass."
      : "Initial vectors share a common phase and representation-dependent uniform magnitude.");
}

function drawMarker(context, x, y, state, selected = false) {
  context.save();
  context.fillStyle = categoryColor(state);
  context.strokeStyle = selected ? COLORS.text : categoryColor(state);
  context.lineWidth = selected ? 2 : 1;
  if (state.optimal) {
    const outer = selected ? 6 : 4.5;
    context.beginPath();
    for (let i = 0; i < 10; i += 1) {
      const angle = -Math.PI / 2 + i * Math.PI / 5;
      const radius = i % 2 === 0 ? outer : outer * 0.45;
      const px = x + radius * Math.cos(angle); const py = y + radius * Math.sin(angle);
      if (i === 0) context.moveTo(px, py); else context.lineTo(px, py);
    }
    context.closePath(); context.fill(); context.stroke();
  } else if (state.feasible) {
    const size = selected ? 7 : 5;
    context.fillRect(x - size / 2, y - size / 2, size, size);
    if (selected) context.strokeRect(x - size / 2, y - size / 2, size, size);
  } else {
    context.beginPath(); context.arc(x, y, selected ? 4.5 : 2.3, 0, Math.PI * 2); context.fill();
    if (selected) context.stroke();
  }
  context.restore();
}

function drawPhaseScatter() {
  const { context, width, height } = setupCanvas(dom.phaseCanvas);
  const positions = appState.run.phase_scatter.display_positions;
  const energies = positions.map((position) => stateAt(position).energy);
  let minEnergy = Math.min(...energies);
  let maxEnergy = Math.max(...energies);
  if (maxEnergy === minEnergy) { minEnergy -= 1; maxEnergy += 1; }
  const plot = drawPlotFrame(context, width, height, { margins: { left: 55, right: 20, top: 24, bottom: 42 } });
  const xFor = (energy) => plot.left + (energy - minEnergy) * (plot.right - plot.left) / (maxEnergy - minEnergy);
  const yFor = (phase) => plot.bottom - (phase + Math.PI) * (plot.bottom - plot.top) / (2 * Math.PI);
  [-Math.PI, -Math.PI / 2, 0, Math.PI / 2, Math.PI].forEach((phase) => {
    const y = yFor(phase);
    context.strokeStyle = COLORS.grid;
    context.beginPath(); context.moveTo(plot.left, y); context.lineTo(plot.right, y); context.stroke();
    const label = phase === 0 ? "0" : (Math.abs(phase) === Math.PI ? `${phase < 0 ? "−" : ""}π` : `${phase < 0 ? "−" : ""}π/2`);
    drawText(context, label, plot.left - 8, y, { align: "right", baseline: "middle" });
  });
  for (let tick = 0; tick <= 4; tick += 1) {
    const energy = minEnergy + tick * (maxEnergy - minEnergy) / 4;
    drawText(context, formatNumber(energy, 0), xFor(energy), plot.bottom + 19, { align: "center", font: "10px Inter, sans-serif" });
  }
  appState.phaseHitRegions = [];
  const sorted = [...positions].sort((left, right) => Number(stateAt(left).feasible) - Number(stateAt(right).feasible));
  sorted.forEach((position) => {
    const state = stateAt(position);
    const value = stateValue(position);
    const x = xFor(state.energy);
    const y = yFor(value[4]);
    drawMarker(context, x, y, state, appState.selectedDisplayPosition === position);
    appState.phaseHitRegions.push({ x, y, position });
  });
  drawText(context, "wrapped phase φ (rad)", plot.left, 14, { color: COLORS.text, font: "11px Inter, sans-serif" });
  drawText(context, "Penalty-QUBO / route energy", (plot.left + plot.right) / 2, height - 7, { align: "center", font: "11px Inter, sans-serif" });
  const scatter = appState.run.phase_scatter;
  dom.phaseSampling.textContent = scatter.downsampled
    ? `${scatter.displayed_count.toLocaleString()} of ${scatter.total_count.toLocaleString()} states; deterministic sample, all feasible retained`
    : `all ${scatter.total_count} logical routes`;
}

function drawEnergyLandscape() {
  const { context, width, height } = setupCanvas(dom.landscapeCanvas);
  const landscape = appState.run.energy_landscape;
  const bins = landscape.bins;
  const plot = drawPlotFrame(context, width, height, { margins: { left: 52, right: 18, top: 28, bottom: 44 } });
  const maxCount = Math.max(...bins.map((bin) => bin.feasible_count + bin.infeasible_count), 1);
  const logMax = Math.log10(maxCount + 1);
  const yFor = (count) => plot.bottom - Math.log10(count + 1) * (plot.bottom - plot.top) / logMax;
  const barWidth = (plot.right - plot.left) / bins.length;
  appState.landscapeHitRegions = [];
  bins.forEach((bin, index) => {
    const x = plot.left + index * barWidth;
    const infeasibleTop = yFor(bin.infeasible_count);
    context.fillStyle = "rgba(255,102,128,0.65)";
    context.fillRect(x + 1, infeasibleTop, Math.max(1, barWidth - 2), plot.bottom - infeasibleTop);
    const feasibleTop = yFor(bin.feasible_count);
    context.fillStyle = COLORS.green;
    context.fillRect(x + 1, feasibleTop, Math.max(1, barWidth - 2), Math.max(0, yFor(0) - feasibleTop));
    appState.landscapeHitRegions.push({ x, width: barWidth, bin });
  });
  landscape.markers.forEach((marker) => {
    const x = plot.left + (marker.energy - landscape.minimum_energy) * (plot.right - plot.left) / (landscape.maximum_energy - landscape.minimum_energy);
    context.strokeStyle = marker.category === "optimal" ? COLORS.gold : (marker.category === "feasible" ? COLORS.green : COLORS.red);
    context.lineWidth = 1.5; context.setLineDash([4, 3]);
    context.beginPath(); context.moveTo(x, plot.top); context.lineTo(x, plot.bottom); context.stroke(); context.setLineDash([]);
    drawText(context, `${marker.label}: ${marker.energy}`, x + 4, plot.top + 12 + landscape.markers.indexOf(marker) * 13, { color: context.strokeStyle, font: "10px Inter, sans-serif" });
  });
  [landscape.minimum_energy, (landscape.minimum_energy + landscape.maximum_energy) / 2, landscape.maximum_energy].forEach((value) => {
    const x = plot.left + (value - landscape.minimum_energy) * (plot.right - plot.left) / (landscape.maximum_energy - landscape.minimum_energy);
    drawText(context, value, x, plot.bottom + 18, { align: "center", font: "10px Inter, sans-serif" });
  });
  drawText(context, "log₁₀(count + 1)", 8, plot.top - 8, { font: "10px Inter, sans-serif" });
  drawText(context, "energy", (plot.left + plot.right) / 2, height - 8, { align: "center" });
  dom.landscapeCounts.textContent = `${landscape.feasible_state_count} feasible / ${landscape.total_state_count.toLocaleString()} total = ${(100 * landscape.feasible_fraction).toFixed(3)}%`;
  dom.landscapeCanvas.onclick = (event) => {
    const rect = dom.landscapeCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const hit = appState.landscapeHitRegions.find((region) => x >= region.x && x < region.x + region.width);
    if (hit) renderEnergyInspector(hit.bin);
  };
}

function renderEnergyInspector(bin) {
  dom.energyInspector.replaceChildren();
  const heading = document.createElement("p");
  heading.textContent = `E ∈ [${formatNumber(bin.left, 2)}, ${formatNumber(bin.right, 2)}): ${bin.feasible_count} feasible, ${bin.infeasible_count.toLocaleString()} infeasible.`;
  dom.energyInspector.append(heading);
  if (!bin.examples.length) return;
  const examples = document.createElement("div");
  examples.className = "energy-examples";
  bin.examples.forEach((state) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `${state.optimal ? "★ " : ""}${state.identity} · E=${state.energy}`;
    button.addEventListener("click", () => selectExternalState(state));
    examples.append(button);
  });
  dom.energyInspector.append(examples);
}

function svgElement(name, attributes = {}) {
  const element = document.createElementNS(SVG_NS, name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
  return element;
}

function renderGeometry() {
  const svg = dom.geometrySvg;
  svg.replaceChildren();
  const title = svgElement("text", { x: 260, y: 24, class: "geometry-label" });
  title.textContent = appState.run.algorithm === "penalty_x"
    ? "Penalty-X: local Hamming-distance-1 mixing"
    : (appState.run.algorithm === "grover_global"
      ? "Global-Grover: rank-one projector over all states"
      : "Feasible-Grover: projector restricted to logical feasible routes");
  svg.append(title);
  const positions = [
    [95, 100], [225, 100], [95, 225], [225, 225],
    [295, 70], [425, 70], [295, 195], [425, 195],
  ];
  if (appState.run.algorithm === "penalty_x") {
    for (let left = 0; left < 8; left += 1) {
      for (let right = left + 1; right < 8; right += 1) {
        if (((left ^ right) & ((left ^ right) - 1)) === 0) {
          svg.append(svgElement("line", { x1: positions[left][0], y1: positions[left][1], x2: positions[right][0], y2: positions[right][1], class: "geometry-edge mixable" }));
        }
      }
    }
  } else {
    const center = [260, 260];
    positions.forEach((position, index) => {
      const feasible = [0, 3, 7].includes(index);
      if (appState.run.algorithm === "grover_global" || feasible) {
        svg.append(svgElement("line", { x1: position[0], y1: position[1], x2: center[0], y2: center[1], class: "geometry-edge mixable" }));
      }
    });
    const projector = svgElement("circle", { cx: center[0], cy: center[1], r: 25, class: "geometry-node projector mixable" });
    svg.append(projector);
    const label = svgElement("text", { x: center[0], y: center[1], class: "geometry-label" });
    label.textContent = appState.run.algorithm === "grover_global" ? "|s_all⟩" : "|s_F⟩";
    svg.append(label);
  }
  positions.forEach((position, index) => {
    const feasible = [0, 3, 7].includes(index);
    const nodeClass = appState.run.algorithm === "grover_feasible" && feasible ? "geometry-node feasible mixable" : "geometry-node mixable";
    const node = svgElement("circle", { cx: position[0], cy: position[1], r: 18, class: nodeClass });
    if (appState.run.algorithm === "grover_feasible" && !feasible) node.setAttribute("opacity", "0.18");
    svg.append(node);
    const label = svgElement("text", { x: position[0], y: position[1], class: "geometry-label" });
    label.textContent = index.toString(2).padStart(3, "0");
    if (appState.run.algorithm === "grover_feasible" && !feasible) label.setAttribute("opacity", "0.23");
    svg.append(label);
  });
  if (appState.run.algorithm !== "penalty_x") {
    const note = svgElement("text", { x: 260, y: 315, class: "geometry-label" });
    note.textContent = "Hub-and-spoke denotes projector action—not a literal complete hardware graph.";
    svg.append(note);
  }
}

function animateGeometry() {
  dom.geometrySvg.querySelectorAll(".mixable").forEach((element) => {
    element.classList.remove("mix-pulse");
    void element.getBoundingClientRect();
    element.classList.add("mix-pulse");
  });
}

function resolveSelectedState() {
  if (appState.selectedDisplayPosition !== null) {
    return { state: stateAt(appState.selectedDisplayPosition), value: stateValue(appState.selectedDisplayPosition) };
  }
  if (appState.selectedExternalState) {
    const external = appState.selectedExternalState;
    const position = appState.run.display_states.findIndex((state) => state.full_state_index === external.full_state_index);
    return { state: external, value: position >= 0 ? stateValue(position) : null };
  }
  const flowState = frame().probability_flow.find((item) => item.kind === "state" && item.category === "optimal")
    || frame().probability_flow.find((item) => item.kind === "state");
  if (flowState) return { state: stateAt(flowState.display_position), value: stateValue(flowState.display_position) };
  return { state: stateAt(0), value: stateValue(0) };
}

function selectDisplayState(position) {
  appState.selectedDisplayPosition = position;
  appState.selectedExternalState = null;
  renderProbabilityFlow(); drawComplexAmplitudes(); drawPhaseScatter(); renderRoute();
}

function selectExternalState(state) {
  const position = appState.run.display_states.findIndex((item) => item.full_state_index === state.full_state_index);
  if (position >= 0) return selectDisplayState(position);
  appState.selectedDisplayPosition = null;
  appState.selectedExternalState = state;
  renderRoute();
}

function renderRoute() {
  const selection = resolveSelectedState();
  const state = selection.state;
  const value = selection.value;
  const graph = appState.run.graph;
  const svg = dom.routeSvg;
  svg.replaceChildren();
  const defs = svgElement("defs");
  const marker = svgElement("marker", { id: "routeArrow", markerWidth: 9, markerHeight: 9, refX: 8, refY: 3, orient: "auto", markerUnits: "strokeWidth" });
  marker.append(svgElement("path", { d: "M0,0 L0,6 L9,3 z", fill: "#839aa0" }));
  defs.append(marker); svg.append(defs);
  const positions = new Map(graph.nodes.map((node) => [node.id, [35 + node.x * 650, 30 + node.y * 310]]));
  const selectedEdges = new Set(state.selected_edge_indices);
  graph.edges.forEach((edge) => {
    const [x1, y1] = positions.get(edge.u); const [x2, y2] = positions.get(edge.v);
    const length = Math.hypot(x2 - x1, y2 - y1);
    const ux = (x2 - x1) / length; const uy = (y2 - y1) / length;
    const line = svgElement("line", {
      x1: x1 + 20 * ux, y1: y1 + 20 * uy, x2: x2 - 23 * ux, y2: y2 - 23 * uy,
      class: `graph-edge${selectedEdges.has(edge.qubit_index) ? " selected" : ""}${state.optimal && selectedEdges.has(edge.qubit_index) ? " optimal" : ""}`,
    });
    svg.append(line);
    const label = svgElement("text", { x: (x1 + x2) / 2 - 7 * uy, y: (y1 + y2) / 2 + 7 * ux, class: "edge-label" });
    label.textContent = `${edge.weight}`;
    svg.append(label);
  });
  const residualByNode = new Map((state.flow_violations || []).map((item) => [item.node, item.residual]));
  graph.nodes.forEach((node) => {
    const [x, y] = positions.get(node.id);
    let nodeClass = "graph-node";
    if (node.id === graph.source) nodeClass += " source";
    if (node.id === graph.target) nodeClass += " target";
    if (residualByNode.has(node.id)) nodeClass += " violation";
    svg.append(svgElement("circle", { cx: x, cy: y, r: 19, class: nodeClass }));
    const label = svgElement("text", { x, y, class: "node-label" }); label.textContent = node.id; svg.append(label);
    if (residualByNode.has(node.id)) {
      const residual = svgElement("text", { x, y: y - 27, class: "residual-label" });
      residual.textContent = `flow ${residualByNode.get(node.id) > 0 ? "+" : ""}${residualByNode.get(node.id)}`; svg.append(residual);
    }
  });
  dom.selectedStateBadge.textContent = `${state.identity} · ${state.optimal ? "globally optimal" : (state.feasible ? "feasible" : "invalid")}`;
  const details = [
    ["identity", state.identity],
    ["probability now", value ? formatProbability(value[3]) : "not included in displayed state sample"],
    ["route / routing term", state.feasible ? `${state.route_text} · cost ${state.route_cost}` : `selected-edge cost ${state.route_cost}`],
    ["total energy", formatNumber(state.energy, 3)],
    ["flow penalty", `${formatNumber(state.flow_penalty, 3)} (A× = ${formatNumber(state.penalty_contribution, 3)})`],
    ["validity", state.validity_explanation, "wide"],
  ];
  dom.routeDetails.replaceChildren();
  details.forEach(([labelText, valueText, className]) => {
    const article = document.createElement("article"); if (className) article.className = className;
    const label = document.createElement("span"); label.textContent = labelText;
    const strong = document.createElement("strong"); strong.textContent = valueText;
    article.append(label, strong); dom.routeDetails.append(article);
  });
}

function colorRamp(value, min, max, invert = false) {
  const t0 = max === min ? 0.5 : (value - min) / (max - min);
  const t = invert ? 1 - t0 : t0;
  const hue = 245 - 195 * Math.max(0, Math.min(1, t));
  return `hsl(${hue} 70% ${36 + 17 * t}%)`;
}

function drawOptimizerLandscape() {
  const { context, width, height } = setupCanvas(dom.optimizerCanvas);
  const landscape = appState.run.parameter_landscape;
  if (!landscape.available) {
    drawText(context, "No scientifically valid p=2 2D slice saved", width / 2, height / 2 - 8, { align: "center", color: COLORS.text, font: "600 15px Inter, sans-serif" });
    drawText(context, "The optimized landscape is four-dimensional; no misleading projection is shown.", width / 2, height / 2 + 20, { align: "center", font: "11px Inter, sans-serif" });
    dom.optimizerDetails.textContent = landscape.reason;
    dom.landscapeMetric.disabled = true;
    return;
  }
  dom.landscapeMetric.disabled = false;
  const values = landscape.metrics[appState.landscapeMetric];
  const min = Math.min(...values); const max = Math.max(...values);
  const plot = drawPlotFrame(context, width, height, { margins: { left: 55, right: 20, top: 24, bottom: 49 } });
  const nGamma = landscape.gammas.length; const nBeta = landscape.betas.length;
  const cellWidth = (plot.right - plot.left) / nBeta;
  const cellHeight = (plot.bottom - plot.top) / nGamma;
  values.forEach((value, index) => {
    const gammaIndex = Math.floor(index / nBeta); const betaIndex = index % nBeta;
    context.fillStyle = colorRamp(value, min, max, appState.landscapeMetric === "expected_hc");
    context.fillRect(plot.left + betaIndex * cellWidth, plot.bottom - (gammaIndex + 1) * cellHeight, cellWidth + 0.5, cellHeight + 0.5);
  });
  const gammaMin = landscape.gammas[0]; const gammaMax = landscape.gammas[landscape.gammas.length - 1];
  const betaMin = landscape.betas[0]; const betaMax = landscape.betas[landscape.betas.length - 1];
  const xFor = (beta) => plot.left + (beta - betaMin) * (plot.right - plot.left) / (betaMax - betaMin);
  const yFor = (gamma) => plot.bottom - (gamma - gammaMin) * (plot.bottom - plot.top) / (gammaMax - gammaMin);
  const final = landscape.final_point; const best = landscape.best_grid_points[appState.landscapeMetric];
  context.strokeStyle = COLORS.text; context.lineWidth = 2;
  const fx = xFor(final.beta); const fy = yFor(final.gamma);
  context.beginPath(); context.moveTo(fx - 7, fy); context.lineTo(fx + 7, fy); context.moveTo(fx, fy - 7); context.lineTo(fx, fy + 7); context.stroke();
  context.strokeStyle = COLORS.gold; context.lineWidth = 2.5; context.beginPath(); context.arc(xFor(best.beta), yFor(best.gamma), 7, 0, Math.PI * 2); context.stroke();
  [gammaMin, (gammaMin + gammaMax) / 2, gammaMax].forEach((value) => drawText(context, formatNumber(value, 2), plot.left - 8, yFor(value), { align: "right", baseline: "middle", font: "10px Inter, sans-serif" }));
  [betaMin, (betaMin + betaMax) / 2, betaMax].forEach((value) => drawText(context, formatNumber(value, 2), xFor(value), plot.bottom + 19, { align: "center", font: "10px Inter, sans-serif" }));
  drawText(context, "γ", 12, plot.top, { color: COLORS.text });
  drawText(context, "β", (plot.left + plot.right) / 2, height - 7, { align: "center", color: COLORS.text });
  drawText(context, "＋ optimized · ○ best grid", plot.left, 14, { color: COLORS.text, font: "10px Inter, sans-serif" });
  const finalMetric = appState.landscapeMetric === "expected_hc" ? appState.run.summary.expected_hc : appState.run.summary[appState.landscapeMetric];
  dom.optimizerDetails.innerHTML = `<strong>Final saved point:</strong> γ=${formatNumber(final.gamma, 5)}, β=${formatNumber(final.beta, 5)}, metric=${formatNumber(finalMetric, 6)}.<br><strong>Best teaching-grid point:</strong> γ=${formatNumber(best.gamma, 5)}, β=${formatNumber(best.beta, 5)}, metric=${formatNumber(best.value, 6)}.<br>${landscape.optimizer_trajectory.reason} ${landscape.teaching_note}`;
}

async function enterPresentation() {
  stopAutoplay();
  appState.presentation = true;
  document.body.classList.add("presentation");
  dom.presentationBar.classList.remove("hidden");
  dom.presentationToggle.textContent = "Presenting";
  await showScene(0);
}

function exitPresentation() {
  appState.presentation = false;
  document.body.classList.remove("presentation");
  dom.presentationBar.classList.add("hidden");
  dom.presentationToggle.textContent = "Present";
  document.querySelectorAll(".panel-section").forEach((panel) => panel.classList.remove("presentation-hidden"));
  renderAll();
}

async function showScene(index) {
  if (!appState.presentation) return;
  const scenes = appState.catalog.presentation_scenes;
  appState.sceneIndex = Math.max(0, Math.min(index, scenes.length - 1));
  const scene = scenes[appState.sceneIndex];
  if (!appState.run || appState.run.algorithm !== scene.algorithm || appState.run.depth !== scene.depth) {
    await loadRun(scene.algorithm, scene.depth, scene.checkpoint);
  } else setCheckpoint(scene.checkpoint);
  dom.sceneCounter.textContent = `Scene ${scene.number} of ${scenes.length}`;
  dom.sceneTitle.textContent = scene.title;
  dom.sceneMessage.textContent = scene.message;
  renderSceneComparison(scene.number);
  dom.scenePrevious.disabled = appState.sceneIndex === 0;
  dom.sceneNext.disabled = appState.sceneIndex === scenes.length - 1;
  document.querySelectorAll(".panel-section").forEach((panel) => panel.classList.toggle("presentation-hidden", !scene.panels.includes(panel.dataset.panel)));
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderSceneComparison(sceneNumber) {
  dom.sceneComparison.replaceChildren();
  let labels = [];
  if (sceneNumber === 4) {
    labels = ["Penalty-X · 16,384 · local X mixer", "Global-Grover · 16,384 · global projector", "same full-space H_C"];
  } else if (sceneNumber === 5) {
    labels = ["Global-Grover · 16,384 bitstrings", "Feasible-Grover · 20 routes", "p_feas=1 is structural"];
  } else if (sceneNumber === 6) {
    labels = appState.catalog.summaries
      .filter((summary) => summary.depth === 2)
      .map((summary) => `${summary.algorithm_short_label}: p_feas ${formatProbability(summary.p_feas)} · p_opt ${formatProbability(summary.p_opt)}`);
  } else if (sceneNumber === 7) {
    labels = [`${appState.run.optimizer.evaluations} evaluations`, appState.run.optimizer.termination, "optimizer trajectory unavailable"];
  }
  labels.forEach((text) => {
    const badge = document.createElement("span");
    badge.textContent = text;
    dom.sceneComparison.append(badge);
  });
}

function showFatal(error) {
  console.error(error);
  dom.errorMessage.textContent = error instanceof Error ? error.message : String(error);
  dom.errorOverlay.classList.remove("hidden");
  dom.loadingOverlay.classList.add("hidden");
}

async function initialize() {
  cacheDom();
  bindEvents();
  try {
    const [catalog, validation] = await Promise.all([fetchJSON("/api/catalog"), fetchJSON("/api/validation")]);
    appState.catalog = catalog;
    if (!validation.all_checks_passed || !catalog.validation.all_checks_passed) throw new Error("Saved-data scientific validation did not pass.");
    dom.validationBadge.textContent = `validated · tol ${validation.tolerance}`;
    dom.validationBadge.classList.add("valid");
    dom.environmentNote.textContent = `Read-only data provenance: ${validation.result_root}. Optimizer rerun: ${validation.optimization_rerun}.`;
    initializeSelectors();
    await loadSelectedRun(0);
    dom.loadingOverlay.classList.add("hidden");
  } catch (error) {
    showFatal(error);
  }
}

document.addEventListener("DOMContentLoaded", initialize);

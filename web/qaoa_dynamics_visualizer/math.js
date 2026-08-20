"use strict";

/* Central semantic notation layer for every user-visible mathematical object.
 * Scientific/API field names remain unchanged; only this layer translates
 * them into presentation notation. KaTeX is vendored locally and loaded first.
 */
(function installQAOAMath(global) {
  const tex = String.raw;

  const FIELD_SEMANTICS = Object.freeze({
    expected_hc: "cost_expectation",
    p_feas: "feasible_probability",
    p_opt: "optimal_probability",
    p_opt_given_feasible: "conditional_optimal_probability",
    invalid_mass: "infeasible_probability",
  });

  const SYMBOLS = Object.freeze({
    cost_expectation: tex`\langle H_C\rangle`,
    feasible_probability: tex`p_{\mathrm{feas}}`,
    optimal_probability: tex`p_{\mathrm{opt}}`,
    conditional_optimal_probability: tex`p_{\mathrm{opt}\mid\mathrm{feas}}`,
    infeasible_probability: tex`p_{\mathrm{infeas}}`,
    probability: tex`P(x)`,
    magnitude: tex`\lvert a_x\rvert`,
    real: tex`\operatorname{Re}(a_x)`,
    imag: tex`\operatorname{Im}(a_x)`,
    phase: tex`\arg(a_x)`,
    energy: tex`E_x`,
    route_cost: tex`C(x)`,
    entropy: tex`S=-\sum_x P(x)\log P(x)`,
    layer: tex`\ell`,
    gamma: tex`\gamma_\ell`,
    beta: tex`\beta_\ell`,
    delta: tex`\Delta`,
  });

  const FORMULAS = Object.freeze({
    layerEvolution: tex`\boxed{\lvert\psi_{\ell-1}\rangle\xrightarrow{\,U_C(\gamma_\ell)\,}\lvert\psi_\ell^{(C)}\rangle\xrightarrow{\,U_M(\beta_\ell)\,}\lvert\psi_\ell\rangle}`,
    fullEvolution: tex`\lvert\psi_p(\boldsymbol{\gamma},\boldsymbol{\beta})\rangle=\prod_{\ell=1}^{p}U_M(\beta_\ell)U_C(\gamma_\ell)\lvert\psi_0\rangle`,
    stateExpansion: tex`\lvert\psi\rangle=\sum_{x\in\{0,1\}^n}a_x\lvert x\rangle`,
    amplitude: tex`a_x=\langle x\rvert\psi\rangle`,
    probability: tex`P(x)=\lvert a_x\rvert^2=\lvert\langle x\rvert\psi\rangle\rvert^2`,
    normalization: tex`\langle\psi\rvert\psi\rangle=\sum_x\lvert a_x\rvert^2=1`,
    costHamiltonian: tex`H_C=c_0I+\sum_j h_jZ_j+\sum_{j<k}J_{jk}Z_jZ_k`,
    basisEnergy: tex`H_C\lvert x\rangle=E_x\lvert x\rangle,\qquad E_x=\langle x\rvert H_C\lvert x\rangle`,
    costUnitary: tex`U_C(\gamma_\ell)=e^{-i\gamma_\ell H_C}`,
    costBasisAction: tex`U_C(\gamma_\ell)\lvert x\rangle=e^{-i\gamma_\ell E_x}\lvert x\rangle`,
    costAmplitudeAction: tex`a_x\longmapsto a_xe^{-i\gamma_\ell E_x},\quad \lvert a_x\rvert^2\longmapsto\lvert a_x\rvert^2,\quad \arg(a_x)\longmapsto\arg(a_x)-\gamma_\ell E_x\pmod{2\pi}`,
    expectation: tex`\langle H_C\rangle_\psi=\langle\psi\rvert H_C\lvert\psi\rangle=\sum_xP(x)E_x`,
    costCommutator: tex`[H_C,U_C(\gamma_\ell)]=0\quad\Longrightarrow\quad\Delta_C\langle H_C\rangle=0`,
    costProbabilityInvariant: tex`\max_x\left\lvert P_x^{\mathrm{after}\ C}-P_x^{\mathrm{before}\ C}\right\rvert\approx0`,
    feasibleProbability: tex`p_{\mathrm{feas}}=\sum_{x\in\mathcal F}\lvert\langle x\rvert\psi\rangle\rvert^2`,
    infeasibleProbability: tex`p_{\mathrm{infeas}}=1-p_{\mathrm{feas}}`,
    optimalProbability: tex`p_{\mathrm{opt}}=\sum_{x\in\mathcal X^\star}\lvert\langle x\rvert\psi\rangle\rvert^2`,
    conditionalProbability: tex`p_{\mathrm{opt}\mid\mathrm{feas}}=\frac{p_{\mathrm{opt}}}{p_{\mathrm{feas}}},\qquad p_{\mathrm{feas}}>0`,
    probabilityDecomposition: tex`p_{\mathrm{opt}}=p_{\mathrm{feas}}\,p_{\mathrm{opt}\mid\mathrm{feas}}`,
    xMixer: tex`H_M^{(X)}=\sum_{j=1}^{n}X_j,\qquad U_M^{(X)}(\beta_\ell)=e^{-i\beta_\ell H_M^{(X)}}`,
    xLocal: tex`X_j=I^{\otimes(j-1)}\otimes X\otimes I^{\otimes(n-j)}`,
    groverMixer: tex`H_M^{(G)}=\Pi_s=\lvert s\rangle\langle s\rvert,\qquad U_M^{(G)}(\beta_\ell)=e^{-i\beta_\ell\Pi_s}=I+\bigl(e^{-i\beta_\ell}-1\bigr)\Pi_s`,
    uniformState: tex`\lvert s\rangle=\frac{1}{\sqrt{2^n}}\sum_{x\in\{0,1\}^n}\lvert x\rangle=\lvert+\rangle^{\otimes n}`,
    plusState: tex`\lvert+\rangle=\frac{\lvert0\rangle+\lvert1\rangle}{\sqrt2}`,
    parameterVectors: tex`\boldsymbol{\gamma}=(\gamma_1,\ldots,\gamma_p),\qquad\boldsymbol{\beta}=(\beta_1,\ldots,\beta_p)`,
    parameterCount: tex`N_{\mathrm{param}}=2p`,
    optimizerObjective: tex`\min_{\boldsymbol{\gamma},\boldsymbol{\beta}}\;\langle\psi_p(\boldsymbol{\gamma},\boldsymbol{\beta})\rvert H_C\lvert\psi_p(\boldsymbol{\gamma},\boldsymbol{\beta})\rangle`,
    routingObjective: tex`C(x)=\sum_{e\in E}w_ex_e`,
    flowPenalty: tex`P_{\mathrm{flow}}(x)=\sum_v\left(\sum_{e\in\delta^+(v)}x_e-\sum_{e\in\delta^-(v)}x_e-b_v\right)^2`,
    routingQubo: tex`Q_A(x)=C(x)+A\,P_{\mathrm{flow}}(x)`,
    binaryToIsing: tex`x_j=\frac{I-Z_j}{2}`,
    mostProbable: tex`x_{\max}=\arg\max_xP(x)`,
    optimum: tex`x^\star\in\arg\min_{x\in\mathcal F}C(x)`,
  });

  function render(element, latex, options = {}) {
    if (!element) throw new Error("math_target_missing");
    if (!global.katex) throw new Error("offline_katex_not_loaded");
    const value = String(latex);
    element.dataset.math = value;
    global.katex.render(value, element, {
      displayMode: Boolean(options.displayMode),
      throwOnError: true,
      strict: "error",
      trust: false,
      output: "htmlAndMathml",
    });
    element.dataset.mathRendered = "true";
    element.classList.add("math-rendered");
    return element;
  }

  function make(latex, options = {}) {
    const element = document.createElement(options.tag || "span");
    if (options.className) element.className = options.className;
    return render(element, latex, options);
  }

  function renderStatic(root = document) {
    root.querySelectorAll("[data-math],[data-math-key]").forEach((element) => {
      if (element.dataset.mathRendered) return;
      const value = element.dataset.mathKey
        ? (SYMBOLS[element.dataset.mathKey] || FORMULAS[element.dataset.mathKey])
        : element.dataset.math;
      if (!value) throw new Error(`unknown_math_semantic:${element.dataset.mathKey}`);
      render(element, value, {
        displayMode: element.dataset.displayMath === "true",
      });
    });
  }

  function symbolForField(field) {
    const semantic = FIELD_SEMANTICS[field];
    if (!semantic || !SYMBOLS[semantic]) throw new Error(`unknown_scientific_field:${field}`);
    return SYMBOLS[semantic];
  }

  function numeric(value, digits = 6, signed = false) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) {
      return tex`\text{--}`;
    }
    const number = Number(value);
    const absolute = Math.abs(number);
    let body;
    if (absolute !== 0 && (absolute < 1e-4 || absolute >= 1e5)) {
      const [mantissa, exponent] = absolute.toExponential(3).split("e");
      body = tex`${mantissa.replace(/0+$/, "").replace(/\.$/, "")}\times10^{${Number(exponent)}}`;
    } else {
      body = absolute.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "") || "0";
    }
    if (number < 0) return `-${body}`;
    return signed ? `+${body}` : body;
  }

  function ket(content) { return tex`\lvert ${content}\rangle`; }
  function gamma(layer) { return tex`\gamma_{${Number(layer)}}`; }
  function beta(layer) { return tex`\beta_{${Number(layer)}}`; }
  function costUnitary(layer) { return tex`U_C\!\left(\gamma_{${Number(layer)}}\right)`; }
  function mixerUnitary(layer, algorithm) {
    const kind = algorithm === "penalty_x" ? "X" : "G";
    return tex`U_M^{(${kind})}\!\left(\beta_{${Number(layer)}}\right)`;
  }
  function stageState(layer, stage) {
    const value = Number(layer);
    if (value === 0) return tex`\lvert\psi_0\rangle`;
    if (stage === "before_cost") return tex`\lvert\psi_{${value - 1}}\rangle`;
    if (stage === "after_cost") return tex`\lvert\psi_{${value}}^{(C)}\rangle`;
    return tex`\lvert\psi_{${value}}\rangle`;
  }
  function currentLayerEvolution(layer, algorithm) {
    const value = Number(layer);
    return tex`\boxed{\lvert\psi_{${value - 1}}\rangle\xrightarrow{\,${costUnitary(value)}\,}\lvert\psi_{${value}}^{(C)}\rangle\xrightarrow{\,${mixerUnitary(value, algorithm)}\,}\lvert\psi_{${value}}\rangle}`;
  }
  function instantiatedGamma(layer, value) {
    return tex`${gamma(layer)}=${Number(value).toFixed(9)}`;
  }
  function instantiatedBeta(layer, value) {
    return tex`${beta(layer)}=${Number(value).toFixed(9)}`;
  }
  function instantiatedCostUnitary(layer, value) {
    return tex`${costUnitary(layer)}=e^{-i(${Number(value).toFixed(9)})H_C}`;
  }
  function instantiatedMixerUnitary(layer, value, algorithm) {
    const kind = algorithm === "penalty_x" ? "X" : "G";
    const generator = algorithm === "penalty_x" ? "H_M^{(X)}" : "\\Pi_s";
    return tex`U_M^{(${kind})}\!\left(\beta_{${Number(layer)}}\right)=e^{-i(${Number(value).toFixed(9)})${generator}}`;
  }
  function transitionPrefix(stage) {
    if (stage === "after_cost") return "C";
    if (stage === "after_mixer") return "M";
    return "";
  }
  function delta(symbol, stage) {
    const prefix = transitionPrefix(stage);
    return prefix ? tex`\Delta_{${prefix}}${symbol}` : tex`\Delta ${symbol}`;
  }
  function stateAmplitude(real, imag) {
    const sign = Number(imag) < 0 ? "-" : "+";
    return tex`a_x=${numeric(real, 6)}${sign}${numeric(Math.abs(Number(imag)), 6)}i`;
  }
  function edgeMarginal(u, v, probability) {
    return tex`\Pr\!\left(x_{(${Number(u)},${Number(v)})}=1\right)=${Number(probability).toFixed(3)}`;
  }
  function hamiltonianTerm(term) {
    if (term.kind === "constant") return "I";
    if (term.kind === "Z") return tex`Z_{${term.qubits[0]}}`;
    return tex`Z_{${term.qubits[0]}}Z_{${term.qubits[1]}}`;
  }

  global.QAOAMath = Object.freeze({
    FIELD_SEMANTICS,
    SYMBOLS,
    FORMULAS,
    render,
    make,
    renderStatic,
    symbolForField,
    numeric,
    ket,
    gamma,
    beta,
    costUnitary,
    mixerUnitary,
    stageState,
    currentLayerEvolution,
    instantiatedGamma,
    instantiatedBeta,
    instantiatedCostUnitary,
    instantiatedMixerUnitary,
    transitionPrefix,
    delta,
    stateAmplitude,
    edgeMarginal,
    hamiltonianTerm,
  });
})(window);

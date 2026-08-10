# Archived three-paper Q2-R design review

## Executive decision

The proposed redesign is scientifically coherent only with a narrower framing than the working title alone suggests.

- Cain et al. diagnose failure of **single-basis-string initialization followed by standard, incumbent-independent QAOA operators**. The present project instead uses a biased product state and an incumbent-dependent aligned mixer. Cain et al. motivate a lock-in investigation, but their negative theorems do not directly apply to this implementation.
- Barkoutsos et al. justify replacing expectation-value optimization by a lower-tail CVaR aggregation for diagonal combinatorial Hamiltonians. This changes the classical loss, not the QAOA circuit. CVaR loss and final success probability must remain separate quantities.
- Saini et al. grow standard QAOA from (p=1) to at most (p=10), transferring parameters by special (p=1\to2) scaling and then interpolation. Their exact algorithm, CSPP encoding, and empirical scaling constants should not be copied. The transferable idea is resource-aware depth growth after an observable plateau.
- Egger et al. define an influential continuous warm-start construction in which the product state is the ground state of a matching separable mixer. The present state-and-mixer pair is mathematically a **binary-incumbent specialization/adaptation** of that construction, but its greedy incumbent is not the paper's continuous-relaxation solution and its (\beta/q) scaling is project-specific.

The resulting study can defensibly ask:

> How do warm-start bias, a tail-focused classical loss, and shallow resource-aware depth growth interact with incumbent escape and route feasibility on one fixed directed weighted routing instance?

It should **not** be presented as the first combination of warm-start QAOA, CVaR, epsilon selection, and depth transfer. Truger et al. already studied all four at fixed incremental depths for MaxCut in 2022. The defensible distinction is the directed flow-penalty routing setting and an explicitly observable, anti-leakage controller for incumbent lock-in and circuit growth.

No implementation or numerical experiment is authorized by this review. The recommended next task is Stage 1 only: add and validate CVaR as an alternative loss while leaving the current circuit, epsilon, depth, optimizer, graph, Hamiltonian, and evaluation oracle unchanged.

## 1. Primary sources and full citations

The review is based on the full author/publisher manuscripts, not titles or secondary summaries.

1. Madelyn Cain, Edward Farhi, Sam Gutmann, Daniel Ranard, and Eugene Tang, “The QAOA gets stuck starting from a good classical string,” arXiv:2207.05089v2 [quant-ph], 7 July 2023 (first submitted 2022). [arXiv](https://arxiv.org/abs/2207.05089), [arXiv DOI](https://doi.org/10.48550/arXiv.2207.05089).
2. Panagiotis Kl. Barkoutsos, Giacomo Nannicini, Anton Robert, Ivano Tavernelli, and Stefan Woerner, “Improving Variational Quantum Optimization using CVaR,” *Quantum* **4**, 256 (2020). [Publisher full text](https://quantum-journal.org/papers/q-2020-04-20-256/), [DOI](https://doi.org/10.22331/q-2020-04-20-256), [arXiv:1907.04769](https://arxiv.org/abs/1907.04769).
3. Rakesh Saini, Nora Mohamed, Saif Al-Kuwari, and Ahmed Farouk, “Dynamic Depth Quantum Approximate Optimization Algorithm for Solving Constrained Shortest Path Problem,” arXiv:2511.08657 [quant-ph], 2025. This is a preprint; no journal publication or journal DOI was identified. [arXiv](https://arxiv.org/abs/2511.08657), [arXiv DOI](https://doi.org/10.48550/arXiv.2511.08657).
4. Daniel J. Egger, Jakub Mareček, and Stefan Woerner, “Warm-starting quantum optimization,” *Quantum* **5**, 479 (2021). [Publisher full text](https://quantum-journal.org/papers/q-2021-06-17-479/), [DOI](https://doi.org/10.22331/q-2021-06-17-479), [arXiv:2009.10095](https://arxiv.org/abs/2009.10095).

Two additional primary sources are important to the novelty assessment:

- Felix Truger, Martin Beisel, Johanna Barzen, Frank Leymann, and Vladimir Yussupov, “Selection and Optimization of Hyperparameters in Warm-Started Quantum Optimization for the MaxCut Problem,” *Electronics* **11**(7), 1033 (2022). [Publisher full text](https://www.mdpi.com/2079-9292/11/7/1033), [DOI](https://doi.org/10.3390/electronics11071033).
- Sean Feeney, Reuben Tate, and Stephan Eidenbenz, “The Better Solution Probability Metric: Optimizing QAOA to Outperform its Warm-Start Solution,” arXiv:2409.09012 [quant-ph] (2024). [arXiv](https://arxiv.org/abs/2409.09012), [arXiv DOI](https://doi.org/10.48550/arXiv.2409.09012).

## 2. Mechanisms extracted from the papers

### 2.1 Cain et al.: what “stuck” means

#### Algorithm actually analyzed

Cain et al. replace the standard initial state (\lvert +\rangle^{\otimes n}) by one computational-basis state (\lvert w\rangle), where (w) is a good classical string, while retaining the standard cost operator (C) and standard X mixer (B=\sum_i X_i). Because (\lvert w\rangle) is an eigenstate of (C), the first cost unitary contributes only a global phase. They therefore use a half-integer depth notation:

\[
\lvert \boldsymbol\gamma,\boldsymbol\beta,w\rangle
=e^{-i\beta_k B}e^{-i\gamma_{k-1}C}\cdots
e^{-i\gamma_1C}e^{-i\beta_1B}\lvert w\rangle,
\qquad p=k-\tfrac12.
\]

The unitary does not explicitly depend on (w); only the optimized scalar angles may depend on it.

#### Characterizations of being stuck

The paper supports several related, but not identical, notions of failure:

1. **No expected-cost improvement.** Numerically, most good starts show zero or tiny improvement after optimizing the QAOA angles. On a 12-vertex 3-regular MaxCut instance, no cut-15 start improved through (p=9/2). On a representative 300-vertex 3-regular instance, 1,000 simulated-annealing starts at each of two temperatures and 1,000 Goemans-Williamson starts showed no improvement at (p=3/2) or (5/2).
2. **Exact shallow obstruction.** At (p=1/2) for MaxCut,

   \[
   \langle C_Z\rangle_\beta=\cos^2(2\beta)C_Z(w).
   \]

   For a good string with (C_Z(w)>0), (\beta=0) is optimal, so the best expected value equals the initial value.
3. **Local small-angle obstruction.** For (p=3/2) on 3-regular MaxCut, let

   \[
   \delta_i=\frac{C(X_iw)-C(w)}{2}.
   \]

   Small-(\beta) improvement is possible if and only if either

   \[
   \sum_{|\delta_i|=1}\delta_i>0
   \quad\text{or}\quad
   \sum_i\delta_i^3>0.
   \]

   A one-bit-flip local maximum has all (\delta_i<0) and satisfies neither condition.
4. **Local thermality at constant depth.** On bounded-degree, mostly locally tree-like regular graphs with edge-local, uniform cost terms, the string's empirical distribution of radius-(p-1/2) neighborhoods is compared with the corresponding thermal marginal. Their thermality coefficient is a trace distance

   \[
   \varepsilon_w=\lVert\rho_{\beta,\mathrm{tree}}-\rho_{w,\mathrm{tree}}\rVert_1.
   \]

   If (\delta) is the fraction of non-tree neighborhoods, the optimized cut fraction obeys

   \[
   \frac1m\langle w\lvert U_w^\dagger C U_w\rvert w\rangle
   \le c(w)+2\varepsilon_w+4\delta.
   \]

   Thus a locally thermal string can improve only slightly at constant depth.
5. **Probability, not only expectation.** Their compression theorem assumes (w) is uniform among the (d_0) strings at cost (C_0), while only (d_1) strings have cost at least (C_1). It bounds the probability that any depth-(p) QAOA circuit samples the target set with probability at least (\epsilon), where

   \[
   \epsilon=
   \left(\frac{2d_1}{d_0}\right)^{1/(2p+2)}
   (16\pi p n^2)^{p/(p+1)}.
   \]

   If (d_1/d_0) is exponentially small and (p=O(n^q)) for (q<1), the improvement/selection probability is very small.

The mechanisms differ by regime. The paper does not establish one universal cause for every warm start. Its evidence combines locality, local optimality, thermal typicality, density-of-states compression, and a fully decoupled example where satisfied terms degrade as much as unsatisfied terms improve.

#### Assumptions and explicit exclusions

The negative result is deliberately limited. It does **not** establish failure of:

- a biased superposition rather than a single basis state;
- a mixer or cost unitary that explicitly depends on the incumbent;
- the Egger or Tate warm-start constructions cited by the authors;
- arbitrary directed weighted routing QUBOs with dense penalty-induced Ising couplings;
- arbitrary depths without the assumptions of the particular theorem;
- an alternative classical loss such as CVaR.

The paper's local-thermality theorem additionally relies on constant depth, bounded degree, mostly tree-like regular graph neighborhoods, and uniform edge-local terms. The compression theorem instead relies on random selection within a cost shell and a sufficiently small density of higher-cost strings. Those assumptions do not silently transfer to this 14-qubit routing Hamiltonian.

#### Meaning for this project

Current Q2 output is empirically incumbent-dominated, so “lock-in” is a useful descriptive diagnosis. It is **not** valid to cite Cain et al. as a theorem that the current algorithm must be stuck. The present initial state has support on every basis string for (0<\epsilon<1/2), and the present mixer explicitly depends on the incumbent. Both facts fall outside the paper's central single-string construction.

### 2.2 Barkoutsos et al.: the exact CVaR convention

#### Minimization and alpha

Let (X(\theta)) be the energy observed by measuring a diagonal combinatorial Hamiltonian in state (\lvert\psi(\theta)\rangle). The paper uses the **lower** tail for minimization. Its (\alpha\in(0,1]) is the fraction of best, lowest-energy outcomes retained:

- (\alpha=1): expectation value;
- smaller (\alpha): stronger focus on low energies;
- (\alpha\downarrow0): best-observed/minimum limit.

This is not the finance convention in which a confidence level near one labels an upper loss tail. Any implementation and plot must say “lower-tail mass (\alpha)” explicitly.

The paper writes

\[
\operatorname{CVaR}_\alpha(X)
=\mathbb E[X\mid X\le F_X^{-1}(\alpha)].
\]

For a discrete distribution, that literal conditional expectation can include more than (\alpha) total mass when an atom lies at the quantile. The intended order-statistic construction retains an (\alpha)-fraction of samples. The published sample equation is also typeset with samples indexed (1,\ldots,K) but a summation beginning at (k=0); this is an off-by-one inconsistency and should not be copied literally.

#### Probability-mass-safe definition for this project

For distinct energies (E_1<E_2<\cdots<E_m) with probabilities (p_j), define

\[
q_\alpha=\min\left\{E_k:\sum_{j\le k}p_j\ge\alpha\right\},
\qquad
s_{<}=\sum_{E_j<q_\alpha}p_j.
\]

The lower-tail loss with **exactly** (\alpha) probability mass is

\[
L_\alpha
=\frac{1}{\alpha}
\left[
\sum_{E_j<q_\alpha}p_jE_j
+(\alpha-s_{<})q_\alpha
\right].
\]

Only the needed fraction of the probability atom at the cutoff is used. This is essential here because many basis states share QUBO energies. It also makes (L_1=\mathbb E[X]) exactly.

For (K) sampled energies sorted as (E_{(1)}\le\cdots\le E_{(K)}), the paper's intended empirical estimator is the average of the first (\lceil\alpha K\rceil) observations. A finite-sample implementation may instead use a fractional last order statistic to represent exactly (\alpha K) empirical mass; if so, it must be documented as a tie/mass-safe refinement rather than attributed verbatim to the paper. Repeated samples at the cutoff are exchangeable, so arbitrary ordering among equal energies has no effect on the energy average.

#### Motivation and supported claims

For a diagonal classical Hamiltonian, the returned combinatorial candidate is the best measured string, not an average-energy quantum observable. The paper argues that a loss emphasizing good outcomes is therefore more aligned with the operational objective than the sample mean, while being smoother than the minimum of a finite sample.

The analytical support is limited but relevant:

- expectation and CVaR landscapes need not share local minima;
- if a state has ground-state probability (\rho>0), it is a global CVaR optimum for every (\alpha\le\rho), whether or not it is an expectation optimum;
- this produces a **soft cap**: once ground-state mass reaches (\alpha), the CVaR loss has no incentive to increase it further;
- the estimator becomes noisier as (\alpha) decreases. The paper reports variance scaling (O(1/(K\alpha^2))), discusses an increased sampling requirement, and empirically recommends the broad range (\alpha\in[0.1,0.25]). These statements do not remove the need to measure variance for this discrete routing spectrum.

Their simulation study covered six combinatorial problem classes, mostly 6–16 qubits, with VQE depths 0–2, QAOA depths 1–3, and (\alpha\in\{0.01,0.05,0.10,0.25,0.50,0.75,1\}). They report faster convergence to higher optimal-state probabilities across their tested instances. Their hardware experiment, however, was CVaR-VQE on one six-qubit portfolio instance, not CVaR-QAOA for routing.

#### Claims not supported

The paper does not establish that CVaR:

- guarantees higher final success probability on a new problem;
- escapes every warm-start incumbent;
- preserves routing feasibility;
- has a universally optimal (\alpha);
- beats state-of-the-art classical heuristics or demonstrates quantum advantage;
- makes CVaR loss values directly comparable across different (\alpha);
- should be used as the final scientific performance metric.

Optimization loss and final success probability are distinct. This project may minimize (L_\alpha), but it must still report (p_{\mathrm{feas}}), incumbent mass, better-feasible-solution probability, and—only in final evaluation—(p_{\mathrm{opt}}).

### 2.3 Saini et al.: dynamic depth and transfer

#### Exact growth rule

DDQAOA begins at (p=1) and grows by one layer at a time up to (p_{\max}=10). Within one Adam optimization loop, it tracks the best expectation energy (E_{\mathrm{best}}). A patience counter is reset only when

\[
E_t<E_{\mathrm{best}}-\epsilon_E.
\]

Depth is increased when either:

- no such improvement has occurred for (k) consecutive iterations; or
- the variance of a recent window (the pseudocode uses roughly the last (k/2) values) falls below a variance threshold (\sigma_E).

The optimizer is reinitialized after a depth increase. The manuscript supplies symbols for the energy tolerance, variance threshold, patience, and maximum optimization iterations, but does not report their numerical experimental values. Its method is therefore not fully reproducible from the paper alone.

#### Exact transfer rule

For (p=1\to2), the paper uses empirical adiabatic-direction factors:

\[
\boldsymbol\gamma^{(2)}=[\gamma_1^{(1)},1.2\gamma_1^{(1)}],
\qquad
\boldsymbol\beta^{(2)}=[\beta_1^{(1)},0.8\beta_1^{(1)}].
\]

For (p\ge2), the old parameters are placed on a uniform normalized layer grid

\[
0,\frac1{p-1},\ldots,1,
\]

and evaluated on the new (p+1)-point grid

\[
0,\frac1p,\ldots,1.
\]

Linear interpolation is used while the old depth is below four; cubic interpolation is used from old depth four onward. Gamma and beta arrays are interpolated separately.

#### Encoding-specific parts

Their experiments use standard QAOA with (\lvert+\rangle^{\otimes N}), an X mixer, expectation loss, and a constrained shortest-path QUBO. The stated CSPP is a bidirected graph with arc cost and one resource consumption. The encoding includes source/sink and flow penalties plus

\[
\rho\left(\sum_{(i,j)}r_{ij}x_{ij}-r_{\mathrm{limit}}\right)^2.
\]

As printed, this square enforces equality to the resource limit, not the stated inequality, unless slack variables or another transformation are supplied; none are shown. That encoding should not replace the DTU flow QUBO.

The paper reports simulated experiments on 100 random instances at the 10- and 16-qubit scales, with DDQAOA growing through (p=10), and fixed-depth baselines (p=3,5,10,15). It uses a PennyLane simulator and Adam, not hardware. Ground-state success probability and an exact-optimum approximation ratio are evaluation metrics, while the expectation is the deployable optimization signal. Reported success probabilities remain small (median 0.017 at 10 qubits and 0.004 at 16 qubits for DDQAOA).

#### What can be borrowed without reproducing DDQAOA

Borrow only the high-level ideas that:

- depth can be treated as a resource allocated after an observable plateau;
- learned shallow parameters are useful initialization data at the next depth;
- cumulative circuit cost should be reported, not only terminal depth.

Do not copy the 1.2/0.8 factors, cubic-switch rule, CSPP QUBO, plus-state initialization, exact convergence pseudocode, or headline comparison. For this project, a neutral appended layer

\[
(\boldsymbol\gamma^{(p+1)},\boldsymbol\beta^{(p+1)})
=((\boldsymbol\gamma^{(p)},0),(\boldsymbol\beta^{(p)},0))
\]

exactly embeds the old circuit at the new depth before reoptimization. This is simpler, testable, and distinct from Saini's interpolation. It is not novel—Truger et al. used the same incremental full-optimization idea—and must be cited accordingly.

### 2.4 Egger et al.: warm-start foundation

#### Continuous warm-start construction

For a continuous relaxation solution (c^*\in[0,1]^n), Egger et al. prepare

\[
\lvert\phi^*\rangle
=\bigotimes_iR_Y(\theta_i)\lvert0\rangle,
\qquad
\theta_i=2\arcsin\sqrt{c_i^*},
\]

so that (P(x_i=1)=c_i^*). They replace the ordinary mixer by

\[
H_{M,i}^{(ws)}=
\begin{pmatrix}
2c_i^*-1 & -2\sqrt{c_i^*(1-c_i^*)}\\
-2\sqrt{c_i^*(1-c_i^*)} & 1-2c_i^*
\end{pmatrix}
=-\sin\theta_iX_i-\cos\theta_iZ_i.
\]

The prepared qubit is its (-1) eigenstate; the product state is the ground state of (H_M^{(ws)}=\sum_iH_{M,i}^{(ws)}). The mixer evolution is implemented with Y/Z/Y rotations.

#### Endpoint regularization

If a relaxation coordinate is exactly zero or one, the matching mixer can freeze that qubit and create a reachability problem. The paper clips coordinates into ([\varepsilon,1-\varepsilon]), with (\varepsilon\in[0,0.5]), and updates both state and mixer. At (\varepsilon=0.5), the construction continuously reaches the standard equal-superposition/X-mixer case. With all coordinates interior, the paper invokes the adiabatic connection to argue asymptotic convergence as (p\to\infty), assuming suitable global parameter schedules/optimization.

The paper also studies rounded warm starts for MaxCut and introduces a sign-flipped mixer variant to recover the rounded Goemans-Williamson cut at selected parameters. That rounded-MaxCut variant is not the same mixer convention used in the present routing project.

#### Relation to the present code

The current project uses

\[
c_i=\begin{cases}1-\varepsilon,&b_i=1,\\\varepsilon,&b_i=0,\end{cases}
\]

for a greedy binary incumbent (b), prepares the same product-state formula, and uses the same aligned Hamiltonian formula. Mathematically, this is the continuous Egger state/mixer construction evaluated on a binary incumbent and regularized away from the poles. It differs in four material ways:

1. (b) is a deterministic greedy feasible route, not the optimum of a continuous QP/SDP relaxation.
2. The code permits only (0<\varepsilon<0.5), not the closed interval.
3. Every mixer angle is scaled by (1/q=1/14), a source-project convention not present in Egger et al.
4. The routing QUBO and directed-flow feasibility structure are project-specific.

The accurate label is therefore **“an incumbent-product adaptation of the Egger aligned-state/aligned-mixer construction.”** Calling the entire current algorithm “the Egger method” would erase the different classical preprocessing and angle convention.

## 3. Mathematical reconstruction of the current project

This section reflects the checked-in source, tests, and results. Where the README's prose might be read more broadly, the executable definitions are authoritative.

### 3.1 Graph and edge encoding

The frozen problem is a directed acyclic graph with (V=\{0,\ldots,6\}), source (s=0), target (t=6), and 14 lexicographically ordered arcs. Qubit (i) represents binary arc variable (x_i).

| (i) | arc | weight | (i) | arc | weight |
|---:|---|---:|---:|---|---:|
| 0 | (0\to1) | 2 | 7 | (2\to4) | 3 |
| 1 | (0\to2) | 4 | 8 | (2\to5) | 7 |
| 2 | (0\to3) | 7 | 9 | (3\to4) | 2 |
| 3 | (1\to2) | 1 | 10 | (3\to5) | 4 |
| 4 | (1\to3) | 4 | 11 | (4\to5) | 2 |
| 5 | (1\to4) | 7 | 12 | (4\to6) | 5 |
| 6 | (2\to3) | 2 | 13 | (5\to6) | 2 |

The computational-basis index is little-endian:

\[
\operatorname{index}(x)=\sum_{i=0}^{13}2^ix_i.
\]

The exact reference enumerates all 20 simple source-to-target paths and independently calls NetworkX shortest path. They agree on the unique optimum (0\to1\to2\to4\to5\to6) of cost 10. This reference is an evaluation oracle, not an input to the greedy warm start.

### 3.2 Flow QUBO

Let (a_{vi}=+1) if arc (i) leaves node (v), (-1) if it enters (v), and zero otherwise. Let (b_s=1), (b_t=-1), and other (b_v=0). The residual and routing cost are

\[
f_v(x)=\sum_i a_{vi}x_i-b_v,
\qquad
C(x)=\sum_iw_ix_i.
\]

The fixed QUBO is

\[
Q_6(x)=C(x)+6\sum_vf_v(x)^2.
\]

In the source convention

\[
Q_6(x)=c+\sum_iq_ix_i+\sum_{i<j}q_{ij}x_ix_j,
\]

the exact coefficients are

\[
c=12,
\]

\[
(q_0,\ldots,q_{13})=
(2,4,7,13,16,19,14,15,19,14,16,14,5,2),
\]

\[
q_{ij}=12\sum_va_{vi}a_{vj}.
\]

Thus nonzero pair coefficients are (+12) for two arcs with the same incidence sign at a node and (-12) for opposite incidence signs. Exhaustive enumeration gives the infeasible/feasible crossing threshold (A_{\mathrm{crit}}=5); (A=6) yields the unique route optimum as the QUBO ground state. Zero flow penalty and the independent “exactly one path, no extra arcs” decoder agree on all (2^{14}) states.

### 3.3 Ising Hamiltonian

With

\[
x_i=\frac{1-Z_i}{2},
\]

the Hamiltonian is

\[
H_C=c_0I+\sum_ih_iZ_i+\sum_{i<j}J_{ij}Z_iZ_j,
\]

where

\[
c_0=86,
\]

\[
(h_0,\ldots,h_{13})=
\left(2,-2,-\tfrac{19}{2},-\tfrac72,-11,-\tfrac{25}{2},-7,
-\tfrac{15}{2},-\tfrac{25}{2},-1,-5,-4,\tfrac12,5\right),
\]

and (J_{ij}=q_{ij}/4\in\{-3,+3\}) for each nonzero QUBO pair. The source exhaustively tests (Q_6(x)=\langle x\lvert H_C\rvert x\rangle) on every basis state.

For optimization the diagonal is affinely normalized:

\[
\widetilde H_C=\frac{H_C-10I}{197},
\]

because the enumerated diagonal ranges from 10 to 207. The identity shift is a global phase and the positive scaling leaves the expectation minimizer unchanged, while changing the numerical interpretation of gamma.

### 3.4 Incumbent and epsilon

The deterministic preprocessing repeatedly takes the lightest outgoing arc that can still reach the target. It returns

\[
b:\quad0\to1\to2\to3\to4\to5\to6,
\qquad C(b)=11.
\]

For (0<\varepsilon<1/2),

\[
c_i(\varepsilon)=
\begin{cases}
1-\varepsilon,&b_i=1,\\
\varepsilon,&b_i=0.
\end{cases}
\]

Epsilon is therefore a per-qubit mismatch probability relative to the incumbent. Small epsilon is stronger bias; increasing epsilon toward (1/2) weakens the bias.

### 3.5 Warm-start state

The prepared state is

\[
\lvert\phi_0(\varepsilon)\rangle
=\bigotimes_{i=0}^{13}
\left(\sqrt{1-c_i}\lvert0\rangle+\sqrt{c_i}\lvert1\rangle\right),
\quad
\theta_i=2\arcsin\sqrt{c_i}.
\]

Hence (P(x_i=1)=c_i). For any basis string (x) at Hamming distance (d_H(x,b)),

\[
P_0(x)=(1-\varepsilon)^{14-d_H(x,b)}\varepsilon^{d_H(x,b)}.
\]

The incumbent has probability ((1-\varepsilon)^{14}). The unique optimum is three bit flips from the incumbent and initially has probability ((1-\varepsilon)^{11}\varepsilon^3). The latter identity is evaluation-only and must not control future epsilon updates.

### 3.6 Mixer: correction to “X mixer”

Q1 uses the ordinary X mixer. Q2 does **not**. For Q2,

\[
H_{M,i}(\varepsilon)
=-2\sqrt{c_i(1-c_i)}X_i-(1-2c_i)Z_i,
\qquad
H_M=\sum_iH_{M,i}.
\]

The prepared qubit is the (-1) eigenstate of (H_{M,i}). This incumbent-dependent aligned mixer is a central reason Cain et al.'s single-string/standard-operator result does not directly cover Q2.

### 3.7 QAOA state and parameter conventions

Parameters are stored as

\[
(\gamma_1,\ldots,\gamma_p,\beta_1,\ldots,\beta_p),
\]

with bounds (\gamma_\ell\in[0,2\pi]) and (\beta_\ell\in[0,\pi]). Each layer applies cost and then mixer. The exact state is

\[
\lvert\psi_p\rangle=
\prod_{\ell=p}^{1}
\left[
e^{-i(\beta_\ell/14)H_M}
e^{-i\gamma_\ell\widetilde H_C}
\right]
\lvert\phi_0(\varepsilon)\rangle,
\]

where the (\ell=1) factor acts first. The (1/14) mixer scaling is project-specific. The generated Qiskit circuit and direct statevector evolution are tested for probability equality.

### 3.8 Current optimization loss

The current classical objective is normalized expected penalized energy:

\[
L_{\mathrm{EE}}(\boldsymbol\gamma,\boldsymbol\beta)
=\langle\psi_p\lvert\widetilde H_C\rvert\psi_p\rangle
=\sum_xP_{\boldsymbol\theta}(x)\widetilde Q_6(x).
\]

Reported energy uses the unnormalized (Q_6). Optimization does not directly maximize feasibility, incumbent escape, or optimal probability.

### 3.9 Current metrics

Let (\mathcal F) be the states accepted by the independent path decoder and (C^*=10) the exact final-evaluation oracle.

\[
p_{\mathrm{feas}}=\sum_{x\in\mathcal F}P(x),
\qquad
p_{\mathrm{opt}}=\sum_{x\in\mathcal F:C(x)=C^*}P(x).
\]

The current project also reports

\[
P_{\mathrm{inc}}=P(x=b),
\]

the feasibility-conditioned expected route cost

\[
\mathbb E[C(X)\mid X\in\mathcal F],
\]

the highest-probability feasible route, optimal-state rank, circuit resources, and initial/final incumbent and optimum probabilities. Because the optimum is unique here, final optimum-state probability equals (p_{\mathrm{opt}}).

All probabilities are computed from the complete exact statevector, not the top-(k) presentation table.

### 3.10 Current optimizer and depth protocol

The baseline uses bounded COBYLA, seed 2601, `rhobeg=0.5`, tolerance (10^{-8}), and at most 100 objective evaluations. Depths are fixed independently at Q1 (p=1), Q2 (p=1), and Q2 (p=2).

The later tuning study:

- sweeps nine epsilons from 0.02 to 0.40 at fixed (p=1,2);
- uses seeds 0–9 and selects the winner within each configuration by lowest expected raw energy;
- tests budgets 100, 200, and 400 for selected epsilons;
- optionally tests small-random angle initialization;
- permits a separate (p=3) study if paired (p=2) results improve both (p_{\mathrm{feas}}) and (p_{\mathrm{opt}}) over (p=1) for at least seven seeds.

It does **not** transfer learned parameters from (p) to (p+1), and depth does not grow inside an optimization run.

#### Anti-leakage audit of existing tuning

Although each same-configuration multistart winner is energy-selected, cross-epsilon ranking and the final recommendation score explicitly use (p_{\mathrm{opt}}) and optimum amplification. The optional (p=3) gate also uses (p_{\mathrm{opt}}). Therefore the checked-in tuned recommendation (\varepsilon=0.15,p=2) is a retrospective, oracle-informed scientific result—not a deployable epsilon/depth policy under the new rule. It may remain in final historical evaluation, but it must not initialize or set thresholds for the new adaptive controller.

## 4. Paper-to-project mapping

| paper | original problem | original algorithm | mechanism to borrow | mechanism explicitly not copied | mathematical compatibility with current implementation | required prospective code changes | scientific risk | distinction from the paper |
|---|---|---|---|---|---|---|---|---|
| Cain et al. | Primarily MaxCut; additional MIS and Sherrington-Kirkpatrick evidence | One computational-basis good string plus standard, incumbent-independent QAOA operators | Motivation and diagnostics for failure to improve an incumbent; distinguish expectation lock from sampling an improved string | Their single-string warm-start circuit, half-integer-depth convention, and any claim that their theorem directly covers Q2 | Limited. Current Q2 has full-support product initialization and an incumbent-dependent mixer, outside their stated scope | Add observable incumbent concentration, better-feasible-solution mass, feasibility, and plateau diagnostics; no circuit replacement | “Lock-in” could become an overclaimed causal label; the fixed routing Hamiltonian violates several theorem assumptions | Directed flow-penalty routing with aligned warm mixer; Cain is diagnosis/motivation only |
| Barkoutsos et al. | Six diagonal combinatorial optimization families; hardware demonstration on six-qubit portfolio VQE | CVaR-VQE and CVaR-QAOA with standard ansätze | Lower-tail CVaR as the classical optimization loss; cutoff-mass-safe exact and sampled estimators | Their VQE ansatz, their problem instances, or a claim that CVaR guarantees success probability | Strong at the loss interface: current exact distribution already supplies energies and probabilities; circuit is unchanged | Implement exact-probability and finite-sample CVaR, alpha metadata, tie/cutoff tests, and optimizer plumbing | Discrete plateaus, soft cap at alpha, higher estimator variance, infeasible low-energy tail mass, alpha overfitting | CVaR is applied inside incumbent-product aligned-mixer WS-QAOA on directed routing, with separate feasibility/escape control |
| Saini et al. | Resource-constrained shortest path on bidirected graphs, 10/16 qubits | Standard plus-state/X-mixer DDQAOA with Adam and expectation loss | Grow depth only after an observable plateau; reuse shallow learned angles; account for cumulative quantum cost | CSPP QUBO, equality-style resource penalty, 1.2/0.8 seed, cubic switch, plus state, exact DDQAOA pseudocode, claimed resource headline | Hamiltonian/parameter dimensions are compatible, but initial state, mixer, optimizer, and loss differ | Add an outer depth controller, neutral-layer transfer, cumulative-evaluation/gate accounting, and stopping logic | Unreported convergence hyperparameters; simulator-only small instances; adaptive methods can receive unequal effective budgets | Warm-start/CVaR/lock-in controller on fixed directed routing; only the general resource-growth idea is borrowed |
| Egger et al. | QUBOs via continuous relaxations; portfolio; rounded MaxCut/RQAOA | Relaxation-derived product state plus matching warm-start mixer; rounded variants | The aligned state/mixer mathematical foundation and endpoint regularization interpretation | Continuous/SDP preprocessing, MaxCut sign-flipped rounded mixer, RQAOA, performance guarantees not inherited by the greedy incumbent | Very strong for state and aligned mixer after setting (c_i^*=b_i) and clipping; different preprocessing and (\beta/14) remain | No foundation-level circuit change is required; documentation and tests must preserve alignment when epsilon changes | Calling the whole method “Egger WS-QAOA” would overstate equivalence; changing state without mixer breaks the construction | Greedy feasible route rather than relaxation solution; directed routing QUBO; project-specific angle normalization |

### 4.1 Risks of copying too closely

- Copying Cain et al.'s basis-state start would replace, rather than diagnose, the project's aligned warm start and would make it a different algorithm.
- Copying Barkoutsos et al.'s displayed discrete conditional formula literally would mishandle atoms at the CVaR cutoff; their VQE circuit and problem instances are also irrelevant here.
- Copying Saini et al.'s 1.2/0.8 scaling, interpolation switch, or convergence controller would be an inadequately reproducible DDQAOA reimplementation rather than a justified transfer to this ansatz.
- Copying Egger et al.'s continuous-relaxation preprocessing would change the classical incumbent mechanism; copying only their state while failing to update the matched mixer when epsilon changes would break the aligned construction.
- Copying Truger et al.'s zero-appended layer transfer without attribution would create a false novelty claim. Here it is retained only as a transparent, exactly neutral initialization for the staged depth experiment.

## 5. Proposed single integrated algorithm

### 5.1 Name and scope

**Lock-in-Aware CVaR Warm-Start QAOA (LA-CVaR-WS-QAOA)** is one Warm-Start QAOA algorithm. It retains:

- one fixed directed graph and edge encoding;
- the fixed (A=6) flow QUBO and its Ising Hamiltonian;
- the greedy incumbent (b);
- the incumbent-product state and its matching aligned mixer;
- the same cost-then-mixer QAOA layers;
- one classical optimizer family.

CVaR changes the classical loss. Epsilon and depth are controlled hyperparameters of that same warm-start ansatz. There is no second QAOA arm, custom path mixer, XY mixer, ADAPT-QAOA, FALQON, QWOA, or GAS.

### 5.2 Pre-registered inputs

The algorithm takes, without consulting the exact optimum:

- fixed lower-tail fraction (\alpha); for the first causal experiment, (\alpha=0.25) is a conservative pre-registered starting choice within Barkoutsos et al.'s empirical 0.1–0.25 range, not a claimed optimum;
- an ordered epsilon schedule (\mathcal E=(\varepsilon_0,\varepsilon_1,\ldots,\varepsilon_{\max})), chosen before final evaluation;
- (p_0=1), a shallow (p_{\max}), total objective-evaluation, shot, and two-qubit-gate budgets;
- patience and uncertainty-aware thresholds for CVaR plateau, incumbent concentration, better-feasible probability, and feasibility retention;
- fixed seeds, angle bounds, optimizer, decoder, and validation policy.

Alpha is fixed during a run. Adapting alpha together with epsilon and depth would make the causal design underidentified.

### 5.3 Observable diagnostics

Every diagnostic is defined without (C^*) or the optimal bitstring.

1. Incumbent probability:

   \[
   q_{\mathrm{inc}}=P(X=b).
   \]

2. Feasibility probability:

   \[
   q_{\mathrm{feas}}=P(X\in\mathcal F).
   \]

3. Better-feasible-solution probability, the routing analogue of Feeney et al.'s BSP:

   \[
   q_{<b}=P(X\in\mathcal F\ \text{and}\ C(X)<C(b)).
   \]

   This requires only the known incumbent cost and route decoder. On this particular graph it happens, after exact analysis, to equal (p_{\mathrm{opt}}), because the only feasible cost below 11 is the unique cost-10 optimum. The controller must nevertheless compute it from the predicate (C(X)<C(b)), never from the optimal string or stored optimum cost.
4. Rolling CVaR improvement and estimator uncertainty.
5. Cumulative objective evaluations, shots, depth, compiled two-qubit gates, and gate-evaluations.

On hardware, probability decisions should use confidence intervals from a fixed diagnostic shot batch. A “no better route” observation is not evidence of zero probability unless its upper confidence bound is below a pre-registered threshold.

### 5.4 Lock-in rule

Declare an **operational lock condition**, not a theorem, only when all hold for the patience window:

- CVaR improvement is smaller than a tolerance tied to numerical/shot uncertainty;
- incumbent concentration is high (preferably measured conditionally as (q_{\mathrm{inc}}/q_{\mathrm{feas}}));
- the upper confidence bound for (q_{<b}) remains below its escape threshold;
- (q_{\mathrm{feas}}) is high enough that modest epsilon relaxation is allowed.

If feasibility is already below its floor, do not relax epsilon merely to reduce incumbent mass. Low incumbent mass caused by spreading probability over infeasible states is not successful escape.

### 5.5 Epsilon adaptation

When the lock condition holds and another pre-registered epsilon level is available:

1. advance exactly one level in (\mathcal E), thereby weakening the incumbent bias;
2. rebuild both the product state **and the aligned mixer** from the new (c_i(\varepsilon));
3. reuse the current gamma/beta vector as the same-depth optimizer initialization;
4. charge all additional evaluations/shots to the same total budget.

Epsilon is never selected by (p_{\mathrm{opt}}), optimum amplification, optimal rank, or exact cost. The incumbent is kept fixed during a run; updating it from samples would create an iterative-warm-start algorithm and confound this design with another mechanism.

### 5.6 Depth growth and parameter transfer

Once the same-depth optimizer has converged and no permitted epsilon action is indicated, grow only if:

- the observable quality target has not been met;
- (p<p_{\max});
- the next layer fits the remaining objective/shot/gate budget; and
- recent observable improvement per incremental resource has not already fallen below the stop threshold.

Transfer by neutral append:

\[
\gamma^{(p+1)}=(\gamma_1^{(p)},\ldots,\gamma_p^{(p)},0),
\qquad
\beta^{(p+1)}=(\beta_1^{(p)},\ldots,\beta_p^{(p)},0).
\]

At the transfer point the new layer is the identity, so the deeper circuit can exactly reproduce the previous state. Reoptimize all (2(p+1)) angles under the remaining budget. This avoids Saini et al.'s unvalidated scale factors and interpolation choices. Because Truger et al. already used zero-appended incremental full optimization, this transfer is a cited engineering choice, not a novelty claim.

### 5.7 Stop and output

Stop on the first applicable condition:

- the lower confidence bound for a pre-registered (q_{<b}) target is met while feasibility remains above its floor;
- CVaR, (q_{<b}), and feasibility show no material improvement over the patience window and neither epsilon nor depth action is justified;
- (p_{\max}), evaluation, shot, or compiled gate-cost budget is exhausted.

Return the best observed feasible route and the full audit trace. Exact (p_{\mathrm{opt}}) and the known optimum are joined only afterward for final evaluation.

### 5.8 Control flow

```text
greedy feasible incumbent b (no exact oracle)
                    |
                    v
aligned warm-start state and mixer at epsilon_0
                    |
                    v
            QAOA at p = 1
                    |
            minimize fixed-alpha CVaR
                    |
                    v
 observable diagnostics: CVaR plateau, q_inc,
       q_<b, q_feas, cumulative resources
          /                |                 \
 lock + safe        quality target met      plateau + budget
    |                       |                 |
    v                       v                 v
next epsilon               stop       neutral transfer p -> p+1
same p, reuse angles                       |
    |                                      |
    +-------------------- optimize <--------+
```

This is one adaptive control loop around one Warm-Start QAOA ansatz.

## 6. Staged causal ablation plan

Turning on CVaR, epsilon adaptation, and depth growth in the first experiment would prevent causal interpretation. Use the following order.

| stage | change from previous stage | fixed-depth comparison | question answered | forbidden adaptation signal |
|---|---|---|---|---|
| 0: current control | None: expectation loss, fixed epsilon, current aligned mixer | Reproduce fixed (p=1) and (p=2) controls | What is the current reference behavior? | All exact-optimum information during optimization/selection |
| 1: CVaR only | Replace expectation loss by one pre-registered fixed-(\alpha) CVaR loss | Pair at the same (p=1,2), same epsilon, seeds, bounds, and evaluation budget | Does tail-focused loss alter incumbent escape or feasibility without circuit changes? | (p_{\mathrm{opt}}), optimal string/cost, alpha chosen by final success |
| 2: lock-aware epsilon only | Enable the observable lock rule and pre-registered epsilon schedule; keep depth fixed, primarily (p=2) | Compare directly with Stage 1 at (p=2) under the same total budget | Does adaptive bias relaxation add benefit beyond CVaR? | Any epsilon update based on exact optimum or retrospective tuned recommendation |
| 3: depth growth/transfer | Add (p\to p+1) growth and neutral transfer | Compare with Stage 2 fixed-depth controls at (p=1,2,3), reporting quality versus cumulative resources | Does adaptive depth add benefit beyond CVaR plus epsilon control? | Exact optimum in growth or stopping; uncharged extra optimizer calls |

Freeze across all stages:

- graph, directed edge order, source/target, incumbent construction, and incumbent bitstring;
- flow QUBO, (A=6), Ising mapping, normalization, and decoder;
- aligned-mixer functional form and (1/14) beta scaling;
- exact-reference implementation and circuit/statevector cross-validation;
- optimizer family, angle bounds, seed set, convergence reporting, and total budget definition;
- final evaluation metrics and deterministic tie handling.

The Stage 2 epsilon value changes by design, but the state/mixer formula is frozen. The Stage 3 depth changes by design, but layer content is frozen.

### Budget fairness

Report both:

1. **Cumulative objective calls/shots**, because adaptive stages perform several optimization episodes.
2. **Cumulative two-qubit gate-evaluations**, approximately

   \[
   \sum_r N_{\mathrm{eval},r}\,G_{2q}(p_r),
   \]

   plus state-preparation and diagnostic-shot costs where relevant.

Terminal circuit depth alone can make a progressive method look artificially cheap. Conversely, forcing equal per-depth iterations gives the adaptive method an artificially large total budget. The primary comparison should use one total resource envelope, with fixed-depth curves as resource-matched references.

### Evaluation outcomes

During adaptation, use only CVaR, (q_{\mathrm{inc}}), (q_{<b}), (q_{\mathrm{feas}}), convergence, and resource cost. After the run is frozen, evaluate:

- (p_{\mathrm{opt}}) and initial/final optimum probability;
- best feasible cost and rank;
- (p_{\mathrm{feas}}), (q_{<b}), and incumbent probability;
- expected raw QUBO energy and CVaR;
- circuit depth, gate counts, total evaluations, and gate-evaluations.

CVaR values at different alpha are not directly comparable; the first ablation therefore fixes alpha.

## 7. Novelty and prior-art risk

### Search scope

Primary-source searches were run across arXiv and publisher records for combinations of:

- “warm-start QAOA” and “CVaR” / “Conditional Value-at-Risk”;
- warm-start/CVaR with shortest path, routing, or vehicle routing;
- dynamic depth, layer-wise/incremental depth, and parameter transfer;
- incumbent improvement / better-solution probability.

An absence result from keyword search is not a proof of first publication. No “first” claim is justified.

### Close prior art

#### Truger et al. (2022): strongest overlap

This paper already evaluates Egger-style WS-QAOA for MaxCut with:

- epsilon values controlling the biased warm start;
- CVaR as one of five classical objective functions;
- “incremental full” depth growth that embeds (p) into (p+1) by adding (\gamma=\beta=0) and reoptimizing all angles;
- “incremental partial” growth that freezes old angles and optimizes only the new layer;
- depths 0 through 3;
- final MaxCut probability and probability of beating the initial cut.

It also tries optimizing epsilon. In their experiments CVaR can improve MaxCut probability relative to expectation, but optimizing epsilon with CVaR sometimes drives epsilon to zero and worsens performance. This is direct evidence that combining the components does not automatically solve lock-in.

Consequences:

- warm-start + CVaR is not new;
- warm-start + CVaR + epsilon study is not new;
- warm-start + CVaR + incremental depth transfer is not new;
- better-than-incumbent probability in this context is not new.

The present work must distinguish itself through directed routing feasibility, its greedy-incumbent aligned mixer and (\beta/14) convention, an uncertainty-aware observable controller, dynamic rather than pre-scheduled growth, and resource-matched staged ablations.

#### Feeney, Tate, and Eidenbenz (2024)

They define Better Solution Probability

\[
\mathrm{BSP}=\sum_{x:C(x)>C(b)}|c_x|^2
\]

for maximization and optimize it directly in aligned-mixer warm-start QAOA on 3-regular MaxCut at (p=1). They explicitly motivate BSP because ground-state probability is unavailable without the optimum. The proposed (q_{<b}) is its minimization/routing analogue. It must be cited as borrowed prior art, not introduced as a new metric.

#### Saini et al. (2025)

They already combine shortest-path structure, dynamic convergence-triggered depth, and parameter transfer. They do not use warm-start state bias, aligned mixers, CVaR, incumbent lock diagnostics, or this DTU encoding. Thus the full DTU question remains different, but “dynamic-depth QAOA for shortest path” cannot be claimed as new.

#### Other nearby directions

Primary literature also contains [warm-start QAOA for vehicle-routing/TSP with XY mixers](https://arxiv.org/abs/2504.19934) and [warm-start VQE studies using CVaR](https://doi.org/10.1140/epjqt/s40507-025-00452-0). Those are outside the mandated algorithm because this project forbids XY/path-exchange mixers and remains QAOA, but they further weaken any broad “warm start plus tail objective for routing is unprecedented” claim.

### Defensibility of the combined question

The question is defensible as a **narrow empirical interaction study and control design**, especially for a course project. It is not a strong component-level algorithmic novelty claim. Its most credible contribution is:

1. reconstructing a precise incumbent-product, aligned-mixer WS-QAOA on a fixed directed flow QUBO;
2. separating incumbent concentration, better-feasible probability, feasibility, CVaR, and exact final success;
3. imposing an anti-leakage runtime controller;
4. performing staged, resource-matched causal ablations.

Because there is only one 14-qubit instance and its optimum is already known to the researchers, conclusions must remain instance-specific. This project can demonstrate interaction on the frozen instance; it cannot establish general routing performance or unseen-instance generalization.

## 8. Scientific and implementation risks

1. **Prior-art compression.** Truger et al. already cover most components. Overbroad novelty language is the largest publication risk.
2. **The Cain mismatch.** Their theorems do not directly apply to the present full-support state and incumbent-dependent mixer.
3. **Oracle contamination.** Existing epsilon/depth recommendations used (p_{\mathrm{opt}}). They cannot seed a clean deployable controller.
4. **Single-instance overfitting.** The graph, optimum, and historical sweeps are all known. Results are hypothesis-generating and instance-specific.
5. **CVaR degeneracy.** Discrete energy atoms create large flat regions. Incorrectly including all cutoff ties changes tail mass and can bias the loss.
6. **CVaR soft cap.** Alpha may limit incentive to increase optimal/better-route mass above alpha.
7. **Feasibility leakage through the loss.** (A=6) guarantees the correct ground state, not that every member of the CVaR tail is feasible. (q_{\mathrm{feas}}) must remain separate.
8. **Adaptive-controller degrees of freedom.** Epsilon schedule, patience, thresholds, alpha, and (p_{\max}) can become a hidden tuning sweep. Pre-register them and expose every trigger.
9. **Resource accounting.** Repeated same-depth reoptimization and progressive depths can cost more than one deep fixed run even if the terminal circuit is shallower.
10. **Optimizer-boundary behavior.** Existing winners often land on gamma/beta bounds. Apparent lock-in or improvement may partly reflect normalization, bounds, or optimizer geometry.
11. **Exact-to-shot mismatch.** Current objectives use exact state probabilities. Hardware/sample CVaR and lock signals require uncertainty handling and can trigger differently.
12. **Parameter-transfer locality.** A neutral appended layer preserves the old state, but COBYLA is not guaranteed to discover a better basin.
13. **Saini reproducibility.** Their numerical convergence thresholds are not reported, and their printed inequality penalty is incomplete without slack handling.

## 9. Recommended next implementation task

Implement **Stage 1: CVaR loss only**, in a separately authorized task.

The task should:

1. add a pure exact-probability lower-tail CVaR function using the fractional cutoff-mass equation in Section 2.2;
2. add a sampled order-statistic estimator with an explicitly documented finite-sample convention;
3. test (\alpha=1) equality with expectation, (\alpha\downarrow0) behavior, degeneracies/ties, fractional cutoff mass, normalization, and invariance to permutations within equal-energy groups;
4. route the loss into the existing optimizer without changing state preparation, mixer, epsilon, depths, graph, QUBO, normalization, or evaluation metrics;
5. run only the pre-registered Stage 1 comparison after a separate authorization;
6. keep (p_{\mathrm{opt}}), exact optimum bitstring/cost, and optimal rank out of optimizer callbacks, alpha choice, seed selection, and winner selection;
7. record expected energy and CVaR simultaneously, but select within a configuration only by its declared loss;
8. defer epsilon adaptation and depth growth until Stage 1 is validated.

This is the smallest change that tests the Barkoutsos mechanism causally and preserves the “one algorithm only” requirement.

## 10. What we must NOT claim

- We invented warm-start QAOA.
- We invented the Egger aligned product-state/mixer construction.
- The complete current project is exactly “the Egger method.”
- We invented CVaR or first applied CVaR to QAOA.
- We first combined warm-start QAOA and CVaR.
- We first combined warm-start QAOA, CVaR, epsilon selection, and incremental depth.
- We invented parameter transfer or zero-layer append.
- We invented dynamic-depth QAOA for shortest path.
- We discovered that a good-string QAOA start can get stuck.
- Cain et al. prove that this current epsilon-biased, aligned-mixer implementation is stuck.
- CVaR optimization is the same thing as maximizing final success probability.
- A lower CVaR value guarantees larger (p_{\mathrm{opt}}) or better feasibility.
- The Saini paper validates this project's encoding, warm start, optimizer, or stopping thresholds.
- The project demonstrates quantum advantage, superiority to classical routing algorithms, scalability, hardware robustness, or general performance on routing.
- The proposed combination is first, unique, or unprecedented.
- The existing (\varepsilon=0.15,p=2) recommendation is an oracle-free deployable policy.
- (p_{\mathrm{opt}}), the exact optimum bitstring, or exact optimum cost influenced epsilon, alpha, depth, stopping, optimizer settings, or run selection in the proposed protocol.

## 11. Source index

### Required papers

- Cain et al.: arXiv:2207.05089; DOI 10.48550/arXiv.2207.05089.
- Barkoutsos et al.: *Quantum* 4, 256; DOI 10.22331/q-2020-04-20-256; arXiv:1907.04769.
- Saini et al.: arXiv:2511.08657; DOI 10.48550/arXiv.2511.08657.
- Egger et al.: *Quantum* 5, 479; DOI 10.22331/q-2021-06-17-479; arXiv:2009.10095.

### Close prior art used in the risk assessment

- Truger et al.: *Electronics* 11(7), 1033; DOI 10.3390/electronics11071033.
- Feeney, Tate, and Eidenbenz: arXiv:2409.09012; DOI 10.48550/arXiv.2409.09012.
- Rafael S. do Carmo et al., “Warm-Starting QAOA with XY Mixers: A Novel Approach for Quantum-Enhanced Vehicle Routing Optimization”: arXiv:2504.19934; DOI 10.48550/arXiv.2504.19934.
- Yahui Chai, Karl Jansen, Stefan Kühn, Tim Schwägerl, and Tobias Stollenwerk, “Warm start of variational quantum algorithms for quadratic unconstrained binary optimization problems,” *EPJ Quantum Technology* (2026); DOI 10.1140/epjqt/s40507-025-00452-0.

### Current-project sources inspected

- `README.md`
- `data/graph.json`, `data/experiment_config.json`
- `src/graph.py`, `src/exact_reference.py`, `src/qubo.py`, `src/ising.py`
- `src/warm_start.py`, `src/qaoa.py`, `src/optimization.py`, `src/metrics.py`
- `src/experiment.py`, `src/tuning.py`
- `scripts/run_experiment.py`, `scripts/tune_warm_start.py`
- all files in `results/baseline/`, `results/tuning/`, `results/final/`
- all files in `tests/`

## 12. Git-status evidence

The repository was already substantially dirty before this authorized task: the task-start snapshot contained 149 tracked/staged status entries belonging to prior work. They were not modified, reverted, committed, or cleaned by this review.

Final scoped status showed:

```text
$ git status --short --untracked-files=all -- docs/literature
?? docs/literature/THREE_PAPER_DESIGN_REVIEW.md
```

The complete status was also compared with the task-start snapshot:

```text
$ wc -l /tmp/sciqis-qaoa-routing.status-before-review /tmp/sciqis-qaoa-routing.status-after-review
149 /tmp/sciqis-qaoa-routing.status-before-review
150 /tmp/sciqis-qaoa-routing.status-after-review

$ comm -13 <(sort /tmp/sciqis-qaoa-routing.status-before-review) <(sort /tmp/sciqis-qaoa-routing.status-after-review)
?? docs/literature/THREE_PAPER_DESIGN_REVIEW.md

$ comm -23 <(sort /tmp/sciqis-qaoa-routing.status-before-review) <(sort /tmp/sciqis-qaoa-routing.status-after-review)
(no output)
```

Thus the only newly introduced status entry is this review document. No source, result, test, configuration, figure, or script file is part of this task's delta.

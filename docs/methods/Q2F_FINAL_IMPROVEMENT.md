# Q2-F final bounded improvement study

## Scope and frozen stopping rule

This is one final, bounded course-project development study in the existing
20-route Q2-F logical space. It is not confirmatory research. It does not
change or rerun the sealed Q2-F study and it does not modify Q2-R.

The success rule was fixed before new results were viewed. A candidate cell
must have median `p_opt >= 0.29139103334`, which is `1.10 * 0.2649009394`, and
every run must satisfy `|p_feas - 1| <= 1e-12`. Among successful cells, the
recommendation uses highest median `p_opt`; candidates within 0.01 absolute of
that value prefer lower depth; a remaining tie uses the frozen simplicity
order BSP path-exchange, GM-QAOA, then GM-Th-QAOA. If no cell succeeds, the
decision is `FREEZE_EXISTING_Q2F`.

## Literature definitions and code mapping

### Better Solution Probability

Feeney, Tate, and Eidenbenz define Better Solution Probability (BSP) as the
total output probability of solutions whose objective value is strictly
better than the supplied warm-start solution. Their maximization expression
uses objective values greater than the warm-start value. They emphasize that
this avoids optimizing ground-state probability, which would require knowing
an optimum in advance. See [The Better Solution Probability Metric: Optimizing
QAOA to Outperform its Warm-Start Solution](https://arxiv.org/abs/2409.09012),
especially Eq. 7.

This routing task is a minimization problem, so the exact literature-to-code
reversal is

\[
 B=\{P_i:C(P_i)<C(P_{\rm incumbent})\},\qquad
 \operatorname{BSP}(\theta)=\sum_{i\in B}|a_i(\theta)|^2,
\]

and bounded COBYLA minimizes

\[
 L_{\rm BSP}(\theta)=-\operatorname{BSP}(\theta).
\]

`better_than_incumbent_mask` accepts only the raw cost vector and incumbent
raw cost. It has no optimum input. V1 changes only the optimizer loss: it
retains the existing Q2-F F1 state (`lambda=1`), ordinary normalized-cost phase,
and genuine logical path-exchange mixer.

### Grover Mixer QAOA

Bärtschi and Eidenbenz define the feasible initial state and rank-one mixer as

\[
 |F\rangle=|F|^{-1/2}\sum_{x\in F}|x\rangle,\qquad
 H_G=|F\rangle\langle F|,
\]

\[
 U_G(\beta)=e^{-i\beta |F\rangle\langle F|}
 =I-(1-e^{-i\beta})|F\rangle\langle F|.
\]

See [Grover Mixers for QAOA: Shifting Complexity from Mixer Design to State
Preparation](https://arxiv.org/abs/2006.00354), Eqs. 2--3. At `beta=pi` this
is the usual Grover diffuser up to a global phase. Its general variational
form is a selective phase shift, not the pre-existing path-exchange adjacency.

V2 implements this operator exactly as a rank-one update in the 20-dimensional
feasible basis. It starts from the uniform feasible state, uses the unchanged
ordinary normalized route-cost phase separator, and minimizes expected
normalized route cost. No state outside the enumerated basis exists.

### Threshold QAOA and GM-Th-QAOA

Golden, Bärtschi, O'Malley, and Eidenbenz replace the continuous objective
phase by a Boolean threshold indicator and combine it with the Grover mixer.
For their maximization convention the indicator is one above a threshold.
See [Threshold-Based Quantum Optimization](https://arxiv.org/abs/2106.13860),
Eqs. 1--3.

For this minimization task, with the prospectively required incumbent
threshold rather than a selected or optimum threshold, V3 uses

\[
 T=C(P_{\rm incumbent}),\qquad
 h_T(P_i)=\begin{cases}
 1,&C(P_i)<T,\\
 0,&C(P_i)\ge T,
 \end{cases}
\]

\[
 U_T(\gamma)=e^{-i\gamma\,\operatorname{diag}(h_T)},\qquad
 U_G(\beta)=e^{-i\beta|F\rangle\langle F|}.
\]

Each of the `p` layers applies `U_T(gamma_l)` and then `U_G(beta_l)`. The
classical loss is `-sum_i h_T(P_i) p_i`. The optimum label is used only after
optimization to report `p_opt`.

This study fixes the threshold to the incumbent cost. The original threshold
paper studies threshold choice as an algorithm parameter, so this variant is
accurately described as an incumbent-threshold GM-Th-QAOA mechanism
demonstration, not a reproduction of its threshold-search protocol.

## Shared bounded matrix

The three variants use fixed depths `p=1,2,3,4`, mandatory seeds
`2601,2602,2603`, and one bounded COBYLA run per cell. Every run has a hard cap
of 150 objective requests, gamma bounds `[0,2*pi]`, beta bounds `[0,pi]`,
`rhobeg=0.5`, and `tol=catol=1e-8`. Parameters are deterministically sampled
with `numpy.random.default_rng(seed)`, all gammas then all betas. There are
exactly 36 runs. No run is replaced or retried because of its outcome.

Raw route costs are preserved. Ordinary phases and the V2 expectation loss use
the existing affine normalization `(C-10)/4`. BSP and threshold membership use
raw costs and the strict incumbent boundary, so normalization cannot change
membership.

## Claim boundary

Full feasible-route enumeration and complete cost-vector construction are
classical preprocessing. `p_feas=1` is structural because all state coordinates
are valid routes; it is not an empirical advantage. GM-Th-QAOA here is a
mechanism demonstration in a tiny logical route space. If the incumbent
threshold happens to leave only one route and that route is the optimum, BSP
will numerically equal `p_opt` on this instance, but the BSP objective was still
constructed without an optimum identity. None of the variants establishes
quantum advantage, hardware performance, or a scalable routing method. Three
seeds do not support a statistical-significance claim.

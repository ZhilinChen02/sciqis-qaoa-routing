"""Penalty-X depth sweep and shared Depth-110 optimization utilities.

The Penalty-X runner uses the frozen ``dynamics_study.StudyContext`` and
layerwise continuation to produce a complete ``p = 1..p_max`` trajectory.
The route-cost Grover-Mixer runner lives in ``depth110_cost_phase.py`` and
imports the shared continuation, COBYLA, row, and serialization helpers from
this module.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version as package_version
from math import pi
from pathlib import Path
import platform
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import Bounds, minimize

from dynamics_study import (
    StudyContext,
    PENALTY_X,
    prepare_study_context,
)
from optimization import EvaluationBudgetExhausted
from qaoa import (
    Q1_PENALTY_X,
    apply_x_mixer,
    simulate_qaoa_state,
    state_probabilities,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "depth_sweep.json"
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results" / "depth110_ext" / "penalty_x_raw"


# ---------------------------------------------------------------------------
# Frozen preflight choices (Section 6.1 / 6.2 of the research plan)
# ---------------------------------------------------------------------------
NEW_LAYER_GAMMA_SCALE = 0.05  # preflight-chosen small-angle scale for new γ_p
NEW_LAYER_BETA_SCALE = 0.05   # preflight-chosen small-angle scale for new β_p
EVAL_BUDGET_BASE = 80         # Track Q B_base
EVAL_BUDGET_PER_PARAM = 1.5   # Track Q k (per parameter count d=2p)
EVAL_BUDGET_MIN = 60          # never starve a shallow depth
EVAL_BUDGET_MAX = 400         # hard ceiling; beyond that the optimizer is wasting time
GAMMA_BOUNDS = (0.0, 2.0 * pi)
BETA_BOUNDS = (0.0, pi)


def eval_budget_for_depth(depth: int) -> int:
    """Track Q budget: B(p) = B_base + k · 2p, clamped to [min, max]."""

    target = EVAL_BUDGET_BASE + int(round(EVAL_BUDGET_PER_PARAM * 2 * depth))
    return min(EVAL_BUDGET_MAX, max(EVAL_BUDGET_MIN, target))


def continuation_initial_parameters(
    previous_optimum: np.ndarray | None,
    *,
    depth: int,
    seed: int,
) -> np.ndarray:
    """Build a depth-p continuation initialization (length exactly 2p).

    p=1: deterministic depth-derived draw — γ_1 ~ U(0, 2π), β_1 ~ U(0, π).
    p>1: inherit θ*_{p-1}, append a fresh seeded small-angle layer
        (γ_p ~ U(0, NEW_LAYER_GAMMA_SCALE), β_p ~ U(0, NEW_LAYER_BETA_SCALE)).

    Each depth's RNG is independently seeded by ``seed + 7919 * depth`` so that
    different depths draw distinct small-angle additions while the overall
    trajectory remains reproducible for a given (seed, depth) pair.
    """

    p = int(depth)
    if p < 1:
        raise ValueError("depth_must_be_positive")
    rng = np.random.default_rng(int(seed) + 7919 * p)
    if previous_optimum is None:
        # Deterministic p=1 draw using the depth-derived RNG seed.
        gamma = float(rng.uniform(0.0, 2.0 * pi))
        beta = float(rng.uniform(0.0, pi))
        return np.asarray([gamma, beta], dtype=np.float64)
    prev = np.asarray(previous_optimum, dtype=np.float64)
    prev_layer_count = prev.shape[0] // 2
    if prev_layer_count > p - 1:
        raise ValueError(
            "continuation_parent_deeper_than_target:"
            f"prev_p={prev_layer_count}, target_p={p}"
        )
    # Pad missing γ's and β's with the canonical small-angle draw.
    if prev_layer_count < p - 1:
        pad_gammas = np.asarray(
            [rng.uniform(0.0, NEW_LAYER_GAMMA_SCALE) for _ in range(p - 1 - prev_layer_count)],
            dtype=np.float64,
        )
        pad_betas = np.asarray(
            [rng.uniform(0.0, NEW_LAYER_BETA_SCALE) for _ in range(p - 1 - prev_layer_count)],
            dtype=np.float64,
        )
    else:
        pad_gammas = np.asarray([], dtype=np.float64)
        pad_betas = np.asarray([], dtype=np.float64)
    prev_gammas = prev[:prev_layer_count]
    prev_betas = prev[prev_layer_count:]
    new_gamma = float(rng.uniform(0.0, NEW_LAYER_GAMMA_SCALE))
    new_beta = float(rng.uniform(0.0, NEW_LAYER_BETA_SCALE))
    return np.concatenate(
        (
            pad_gammas,
            prev_gammas,
            np.asarray([new_gamma], dtype=np.float64),
            pad_betas,
            prev_betas,
            np.asarray([new_beta], dtype=np.float64),
        )
    ).astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# COBYLA-based optimizer mirror accepting explicit x0 and arbitrary depth
# (used by both algorithms; the loss function and param order are caller-set)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class COBYLAOutcome:
    parameters: tuple[float, ...]
    initial_objective: float
    final_objective: float
    evaluations: int
    success: bool
    reason: str
    wall_time: float


def cobyla_with_explicit_x0(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    depth: int,
    evaluation_budget: int,
    gamma_bounds: Sequence[float] = GAMMA_BOUNDS,
    beta_bounds: Sequence[float] = BETA_BOUNDS,
    rhobeg: float = 0.5,
    tolerance: float = 1e-8,
) -> COBYLAOutcome:
    """Bounded COBYLA identical in semantics to ``optimize_cobyla``,
    but the initial point is supplied by the caller (continuation)."""

    p, budget = int(depth), int(evaluation_budget)
    if p < 1 or budget < 1:
        raise ValueError("invalid_optimizer_depth_or_budget")
    if np.asarray(x0, dtype=np.float64).shape != (2 * p,):
        raise ValueError("explicit_x0_length_mismatch")
    lower = np.asarray(
        [float(gamma_bounds[0])] * p + [float(beta_bounds[0])] * p, dtype=np.float64
    )
    upper = np.asarray(
        [float(gamma_bounds[1])] * p + [float(beta_bounds[1])] * p, dtype=np.float64
    )
    evaluations = 0
    best_value = float("inf")
    best_parameters = np.asarray(x0, dtype=np.float64).copy()
    initial_value: float | None = None

    def counted(parameters: np.ndarray) -> float:
        nonlocal evaluations, best_value, best_parameters, initial_value
        if evaluations >= budget:
            raise EvaluationBudgetExhausted
        evaluations += 1
        candidate = np.asarray(parameters, dtype=np.float64)
        outside = float(
            np.sum(np.maximum(lower - candidate, 0.0))
            + np.sum(np.maximum(candidate - upper, 0.0))
        )
        value = 1_000_000.0 + outside if outside else float(objective(candidate))
        if not np.isfinite(value):
            raise ValueError("non_finite_optimizer_objective")
        if initial_value is None:
            initial_value = value
        if not outside and value < best_value:
            best_value = value
            best_parameters = candidate.copy()
        return value

    started = perf_counter()
    scipy_result = None
    exhausted = False
    try:
        scipy_result = minimize(
            counted,
            best_parameters.copy(),
            method="COBYLA",
            bounds=Bounds(lower, upper),
            options={
                "maxiter": budget,
                "rhobeg": float(rhobeg),
                "tol": float(tolerance),
                "catol": float(tolerance),
            },
        )
    except EvaluationBudgetExhausted:
        exhausted = True
    wall_time = perf_counter() - started
    if evaluations == 0 or initial_value is None:
        raise RuntimeError("optimizer_performed_no_evaluations")
    if scipy_result is not None:
        candidate = np.asarray(scipy_result.x, dtype=np.float64)
        in_bounds = bool(np.all(candidate >= lower) and np.all(candidate <= upper))
        if in_bounds and float(scipy_result.fun) < best_value:
            best_value = float(scipy_result.fun)
            best_parameters = candidate
    success = bool(scipy_result is not None and scipy_result.success and not exhausted)
    if exhausted or evaluations >= budget:
        reason = "evaluation_budget_exhausted"
    elif success:
        reason = "converged"
    else:
        reason = f"optimizer_failed:{getattr(scipy_result, 'message', 'unknown_error')}"
    return COBYLAOutcome(
        parameters=tuple(map(float, best_parameters)),
        initial_objective=float(initial_value),
        final_objective=float(best_value),
        evaluations=int(evaluations),
        success=bool(success),
        reason=str(reason),
        wall_time=float(wall_time),
    )


# ---------------------------------------------------------------------------
# Penalty-X driver (full 14-qubit space, 2^14 = 16384-dim statevector)
# ---------------------------------------------------------------------------
def _x_mixer_evolve(state: Sequence[complex], beta: float) -> np.ndarray:
    """Same course parameter convention as ``dynamics_study._x_mixer_evolve``:
    the COBYLA parameter β_qa is rescaled to ``beta / 14`` before applying the
    separable X mixer to the 14-qubit circuit."""
    return apply_x_mixer(
        np.asarray(state, dtype=np.complex128),
        float(beta) / 14.0,
        14,
    )


def _penalty_x_objective(context: StudyContext, parameters: np.ndarray, depth: int) -> float:
    state = simulate_qaoa_state(
        context.normalized_full_diagonal,
        parameters,
        depth=int(depth),
        solver=Q1_PENALTY_X,
    )
    probabilities = state_probabilities(state)
    return float(probabilities @ context.normalized_full_diagonal)


def _evaluate_penalty_x_final(context: StudyContext, parameters: np.ndarray, depth: int) -> dict[str, Any]:
    state = simulate_qaoa_state(
        context.normalized_full_diagonal,
        parameters,
        depth=int(depth),
        solver=Q1_PENALTY_X,
    )
    probabilities = state_probabilities(state)
    feasible = np.asarray(context.full_metadata.feasible_mask, dtype=bool)
    optimal = np.asarray(context.full_metadata.optimal_mask, dtype=bool)
    route_costs = np.asarray(context.full_metadata.route_costs, dtype=np.float64)
    p_feas = float(np.sum(probabilities[feasible]))
    p_opt = float(np.sum(probabilities[optimal]))
    invalid_mass = float(np.sum(probabilities[~feasible]))
    expected_cost_full = float(probabilities @ route_costs)
    expected_cost_feasible = (
        float(probabilities[feasible] @ route_costs[feasible] / p_feas)
        if p_feas > 1e-15
        else float("nan")
    )
    feasible_conditional_p_opt = (
        float(probabilities[optimal].sum() / p_feas) if p_feas > 1e-15 else float("nan")
    )
    return {
        "p_feas": p_feas,
        "p_opt": p_opt,
        "invalid_mass": invalid_mass,
        "expected_cost_full": expected_cost_full,
        "expected_cost_feasible": expected_cost_feasible,
        "feasible_conditional_p_opt": feasible_conditional_p_opt,
        "probabilities": probabilities,
    }


# ---------------------------------------------------------------------------
# Depth-chain driver
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DepthRow:
    algorithm: str
    depth: int
    seed: int
    representation: str
    search_dimension: int
    mixer_type: str
    parameter_count: int
    initialization_source: str
    evaluation_budget: int
    evaluation_budget_used: int
    optimizer_reason: str
    optimizer_success: bool
    optimizer_wall_time_seconds: float
    p_feas: float
    p_opt: float
    invalid_mass: float
    expected_cost_full: float
    expected_cost_feasible: float
    feasible_conditional_p_opt: float
    normalized_regret: float
    bsp: float
    parent_depth_parameters: tuple[float, ...] | None
    final_parameters: tuple[float, ...]

    def as_csv_row(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parent_depth_parameters"] = (
            None if self.parent_depth_parameters is None
            else list(self.parent_depth_parameters)
        )
        payload["final_parameters"] = list(self.final_parameters)
        return payload


def run_penalty_x_chain(
    context: StudyContext,
    *,
    depths: Sequence[int],
    seed: int,
) -> list[DepthRow]:
    rows: list[DepthRow] = []
    prev_optimum: np.ndarray | None = None
    for depth in depths:
        x0 = continuation_initial_parameters(prev_optimum, depth=depth, seed=seed)
        budget = eval_budget_for_depth(depth)
        outcome = cobyla_with_explicit_x0(
            lambda p: _penalty_x_objective(context, p, depth),
            x0,
            depth=depth,
            evaluation_budget=budget,
        )
        diag = _evaluate_penalty_x_final(context, np.asarray(outcome.parameters), depth)
        exact_cost = float(context.exact_cost)
        regret = (diag["expected_cost_full"] - exact_cost) / max(exact_cost, 1e-12)
        rows.append(
            DepthRow(
                algorithm=PENALTY_X,
                depth=int(depth),
                seed=int(seed),
                representation="14 edge bits; all 2^14 bitstrings",
                search_dimension=int(context.full_metadata.dimension),
                mixer_type="H_X=sum_j X_j (course beta/q scaling)",
                parameter_count=2 * int(depth),
                initialization_source=(
                    "depth_derived_seed" if depth == 1
                    else f"continuation_from_p{depth - 1}_plus_seed_small_angle"
                ),
                evaluation_budget=budget,
                evaluation_budget_used=outcome.evaluations,
                optimizer_reason=outcome.reason,
                optimizer_success=outcome.success,
                optimizer_wall_time_seconds=outcome.wall_time,
                p_feas=diag["p_feas"],
                p_opt=diag["p_opt"],
                invalid_mass=diag["invalid_mass"],
                expected_cost_full=diag["expected_cost_full"],
                expected_cost_feasible=diag["expected_cost_feasible"],
                feasible_conditional_p_opt=diag["feasible_conditional_p_opt"],
                normalized_regret=regret,
                bsp=float("nan"),
                parent_depth_parameters=(
                    None if prev_optimum is None
                    else tuple(map(float, prev_optimum))
                ),
                final_parameters=tuple(outcome.parameters),
            )
        )
        prev_optimum = np.asarray(outcome.parameters, dtype=np.float64)
        print(
            f"  [penalty_x]   p={depth:3d}  p_opt={diag['p_opt']:.6f}  "
            f"p_feas={diag['p_feas']:.6f}  evals={outcome.evaluations:3d}/{budget}  "
            f"t={outcome.wall_time:6.2f}s  {outcome.reason}"
        )
    return rows


# ---------------------------------------------------------------------------
# Failure / row schema / serialization
# ---------------------------------------------------------------------------
def attach_failure_tags(rows: Sequence[DepthRow]) -> list[dict[str, str]]:
    """Apply the failure taxonomy (Section 12.1 of the plan) to each row."""

    tags: list[dict[str, str]] = []
    for previous, row in zip([None, *rows[:-1]], rows):
        status = "COMPLETED"
        if row.optimizer_reason.startswith("evaluation_budget_exhausted"):
            status = "BUDGET_LIMITED"
        elif row.optimizer_reason.startswith("optimizer_failed"):
            status = "OPTIMIZER_FAILED"
        if previous is not None and status == "COMPLETED":
            delta_p_opt = row.p_opt - previous.p_opt
            if delta_p_opt < -1e-4 and abs(row.p_opt - previous.p_opt) > 0.005:
                status = "REGRESSION"
            elif abs(delta_p_opt) < 1e-9:
                status = "NO_IMPROVEMENT"
        tags.append(
            {
                "algorithm": row.algorithm,
                "depth": str(row.depth),
                "status": status,
                "optimizer_reason": row.optimizer_reason,
            }
        )
    return tags


def _csv_fieldnames() -> list[str]:
    return [
        "algorithm",
        "depth",
        "seed",
        "representation",
        "search_dimension",
        "mixer_type",
        "parameter_count",
        "initialization_source",
        "evaluation_budget",
        "evaluation_budget_used",
        "optimizer_reason",
        "optimizer_success",
        "optimizer_wall_time_seconds",
        "p_feas",
        "p_opt",
        "invalid_mass",
        "expected_cost_full",
        "expected_cost_feasible",
        "feasible_conditional_p_opt",
        "normalized_regret",
        "bsp",
    ]


def write_canonical_rows(rows: Sequence[DepthRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _csv_fieldnames()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: getattr(row, key) for key in fieldnames})


def write_parameters(rows: Sequence[DepthRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        f"{row.algorithm}_p{int(row.depth)}_seed{int(row.seed)}": {
            "parameters": list(row.final_parameters),
            "parent_parameters": (
                None if row.parent_depth_parameters is None
                else list(row.parent_depth_parameters)
            ),
            "depth": int(row.depth),
            "algorithm": row.algorithm,
        }
        for row in rows
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_runtime_profile(rows: Sequence[DepthRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "algorithm", "depth", "optimizer_wall_time_seconds",
        "evaluation_budget_used", "optimizer_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: getattr(row, key) for key in fieldnames})


def write_environment(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "numpy": package_version("numpy"),
                "scipy": package_version("scipy"),
                "matplotlib": package_version("matplotlib"),
                "qiskit": package_version("qiskit"),
                "optimizer": "scipy.optimize.minimize(method=COBYLA)",
                "statevector_mode": "ideal_exact_numpy_complex128",
                "continuation_rule": "theta_p_init = (theta*_{p-1}, gamma_p_small, beta_p_small)",
                "track_q_budget_formula": "B(p) = 80 + 1.5 * 2p, clamped to [60, 400]",
                "depth_range_frozen": "1..110 inclusive",
                "algorithms_frozen": [PENALTY_X],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_experiment_config(
    *,
    penalty: float,
    seed: int,
    depths: Sequence[int],
    graph_path: str,
    graph_sha256: str,
    path: Path,
) -> None:
    payload = {
        "schema": "dtu-sciqis-qaoa-depth110-extension",
        "version": "1.0",
        "title": "Depth-110 Penalty-X QAOA Source Sweep",
        "algorithms": [PENALTY_X],
        "depths": [int(d) for d in depths],
        "base_seed": int(seed),
        "depth_rng_seed_formula": "base_seed + 7919 * p",
        "optimizer": "COBYLA",
        "objective_evaluation_cap_formula": (
            f"B(p) = {EVAL_BUDGET_BASE} + {EVAL_BUDGET_PER_PARAM} * 2p, "
            f"clamped to [{EVAL_BUDGET_MIN}, {EVAL_BUDGET_MAX}]"
        ),
        "continuation_rule": {
            "p_equal_1": (
                "deterministic depth-derived initialization from base seed; "
                "depth RNG seed = base_seed + 7919 * p"
            ),
            "p_greater_1": (
                "inherit previous optimum; new gamma_p ~ U(0, "
                f"{NEW_LAYER_GAMMA_SCALE}); new beta_p ~ U(0, {NEW_LAYER_BETA_SCALE})"
            ),
        },
        "gamma_bounds": list(GAMMA_BOUNDS),
        "beta_bounds": list(BETA_BOUNDS),
        "penalty": float(penalty),
        "graph_path": graph_path,
        "graph_sha256": graph_sha256,
        "claim_quantum_advantage": False,
        "claim_scalable_routing": False,
        "plan_reference": "project research plan, Penalty-X source trajectory",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# High-level sweep driver
# ---------------------------------------------------------------------------
def run_penalty_x_depth_sweep(
    *,
    depths: Sequence[int] = tuple(range(1, 111)),
    seed: int = 2601,
    penalty: float = 6.0,
    result_root: Path = DEFAULT_RESULT_ROOT,
    save_distributions: bool = True,
) -> dict[str, Any]:
    """Run the Penalty-X depth chain and persist canonical rows, parameters,
    runtime profile, failure labels, and optional final distributions.
    Returns a summary dict for the calling CLI."""

    print(f"[penalty_x_sweep] Building study context (penalty={penalty})...")
    context = prepare_study_context(penalty=float(penalty))
    print(
        f"[penalty_x_sweep]   full dim={context.full_metadata.dimension}, "
        f"feasible dim={context.feasible_metadata.dimension}, "
        f"exact route={context.exact_route}, exact_cost={context.exact_cost}"
    )
    graph_path = str(Path("data") / "graph.json")
    graph_sha256 = sha256((PROJECT_ROOT / "data" / "graph.json").read_bytes()).hexdigest()

    result_root = Path(result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    write_environment(result_root / "environment.json")
    write_experiment_config(
        penalty=penalty,
        seed=seed,
        depths=depths,
        graph_path=graph_path,
        graph_sha256=graph_sha256,
        path=result_root / "experiment_config.json",
    )

    all_rows: list[DepthRow] = []
    print(f"[penalty_x_sweep] p=1..{max(depths)} with layerwise continuation (base seed={seed})")
    all_rows.extend(run_penalty_x_chain(context, depths=list(depths), seed=seed))

    write_canonical_rows(all_rows, result_root / "canonical_rows.csv")
    write_parameters(all_rows, result_root / "optimized_parameters.json")
    write_runtime_profile(all_rows, result_root / "runtime_profile.csv")

    failures = attach_failure_tags(all_rows)
    (result_root / "failure_taxonomy.json").write_text(
        json.dumps(failures, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if save_distributions:
        save_final_distributions(all_rows, result_root / "final_distributions.npz")

    return {
        "result_root": str(result_root),
        "row_count": len(all_rows),
        "depth_range": [int(min(depths)), int(max(depths))],
        "failures": sum(1 for f in failures if f["status"] != "COMPLETED"),
    }


def save_final_distributions(rows: Sequence[DepthRow], path: Path) -> None:
    """Save per-depth final probability vectors for forensic figure regen.

    Uses a single shared ``StudyContext`` rebuild (cached in a module global).
    """

    ctx = _rebuild_context_for_rows()
    payload: dict[str, np.ndarray] = {}
    for row in rows:
        parameters = np.asarray(row.final_parameters, dtype=np.float64)
        if row.algorithm != PENALTY_X:
            raise ValueError(f"penalty_x_distribution_writer_received:{row.algorithm}")
        diag = _evaluate_penalty_x_final(ctx, parameters, int(row.depth))
        probs = diag["probabilities"]
        payload[f"{row.algorithm}_p{int(row.depth)}_seed{int(row.seed)}"] = probs
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


_REBUILD_CONTEXT: StudyContext | None = None
def _rebuild_context_for_rows() -> StudyContext:
    global _REBUILD_CONTEXT
    if _REBUILD_CONTEXT is None:
        _REBUILD_CONTEXT = prepare_study_context(penalty=6.0)
    return _REBUILD_CONTEXT


if __name__ == "__main__":  # pragma: no cover - convenience local smoke entry
    import argparse

    parser = argparse.ArgumentParser(description="Run the Penalty-X depth sweep")
    parser.add_argument("--max-depth", type=int, default=110)
    parser.add_argument("--seed", type=int, default=2601)
    parser.add_argument(
        "--depths",
        type=int,
        nargs="*",
        default=None,
        help="optional explicit depth list; overrides --max-depth",
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument(
        "--no-distributions",
        action="store_true",
        help="skip saving per-depth final probability vectors",
    )
    args = parser.parse_args()
    depths = args.depths if args.depths else tuple(range(1, args.max_depth + 1))
    summary = run_penalty_x_depth_sweep(
        depths=depths,
        seed=args.seed,
        result_root=args.result_root,
        save_distributions=not args.no_distributions,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

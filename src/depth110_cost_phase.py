"""Cost-phase Grover-Mixer depth sweep for the Depth-110 main track.

The feasible basis and rank-one Grover mixer are reused from the course
project.  The cost layer is the diagonal route-cost operator

    H_C = sum_i C(P_i) |P_i><P_i|,

so each feasible route receives the phase exp(-i * gamma * C(P_i)).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from depth_sweep import (
    BETA_BOUNDS,
    EVAL_BUDGET_BASE,
    EVAL_BUDGET_MAX,
    EVAL_BUDGET_MIN,
    EVAL_BUDGET_PER_PARAM,
    GAMMA_BOUNDS,
    NEW_LAYER_BETA_SCALE,
    NEW_LAYER_GAMMA_SCALE,
    DepthRow,
    cobyla_with_explicit_x0,
    continuation_initial_parameters,
    eval_budget_for_depth,
    write_canonical_rows,
    write_parameters,
    write_runtime_profile,
)
from dynamics_study import StudyContext, prepare_study_context
from q2f_final_improvement import build_incumbent_threshold


GROVER_COST_PHASE = "grover_cost_phase"


def cost_phase_setup(
    context: StudyContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return route costs, the incumbent mask, and normalized costs.

    Only ``route_cost_phase`` enters the QAOA cost evolution.  The marked mask
    is retained solely for the auxiliary better-solution probability ``bsp``.
    """

    route_cost_phase = np.asarray(
        context.feasible_metadata.route_costs,
        dtype=np.float64,
    )
    incumbent_cost = float(route_cost_phase[context.basis.incumbent_route_id])
    threshold = build_incumbent_threshold(route_cost_phase, incumbent_cost)
    marked = np.asarray(threshold.better_mask, dtype=bool)
    normalized_costs = np.asarray(
        context.normalized_feasible_diagonal,
        dtype=np.float64,
    )

    if route_cost_phase.shape != (context.feasible_metadata.dimension,):
        raise ValueError("route_cost_phase_dimension_mismatch")
    if not np.isfinite(route_cost_phase).all():
        raise ValueError("route_cost_phase_contains_non_finite_values")
    if np.unique(route_cost_phase).size < 2:
        raise ValueError("route_cost_phase_must_contain_multiple_costs")

    return route_cost_phase, marked, normalized_costs


def simulate_cost_phase_grover_state(
    initial_state: np.ndarray,
    route_cost_phase: np.ndarray,
    mixer,
    parameters: np.ndarray,
    *,
    depth: int,
) -> np.ndarray:
    """Apply p alternating route-cost and feasible-Grover layers."""

    p = int(depth)
    values = np.asarray(parameters, dtype=np.float64)
    phase = np.asarray(route_cost_phase, dtype=np.float64)
    state = np.asarray(initial_state, dtype=np.complex128).copy()

    if p < 1:
        raise ValueError("depth_must_be_positive")
    if values.shape != (2 * p,):
        raise ValueError("grover_cost_phase_param_count_mismatch")
    if state.shape != (mixer.dimension,) or phase.shape != state.shape:
        raise ValueError("grover_cost_phase_dimension_mismatch")
    if abs(float(np.linalg.norm(state)) - 1.0) > 1e-9:
        raise ValueError("grover_cost_phase_initial_state_unnormalized")

    gammas = values[:p]
    betas = values[p:]
    for layer in range(p):
        state = state * np.exp(-1j * float(gammas[layer]) * phase)
        if abs(float(np.linalg.norm(state)) - 1.0) > 1e-9:
            raise RuntimeError("grover_cost_phase_cost_layer_changed_norm")
        state = mixer.evolve(state, float(betas[layer]))
        if abs(float(np.linalg.norm(state)) - 1.0) > 1e-9:
            raise RuntimeError("grover_cost_phase_mixer_changed_norm")

    return state


def evaluate_cost_phase_grover(
    context: StudyContext,
    parameters: np.ndarray,
    depth: int,
    *,
    route_cost_phase: np.ndarray,
    marked: np.ndarray,
) -> dict[str, Any]:
    """Return the main metrics and final probability vector."""

    state = simulate_cost_phase_grover_state(
        np.asarray(context.feasible_initial_state, dtype=np.complex128),
        route_cost_phase,
        context.feasible_mixer,
        parameters,
        depth=int(depth),
    )
    probabilities = np.abs(state) ** 2
    total = float(probabilities.sum())
    if abs(total - 1.0) > 1e-9:
        raise RuntimeError(f"grover_cost_phase_final_state_unnormalized:{total}")
    probabilities = probabilities / total

    route_costs = np.asarray(
        context.feasible_metadata.route_costs,
        dtype=np.float64,
    )
    optimal_mask = np.asarray(
        context.feasible_metadata.optimal_mask,
        dtype=bool,
    )
    expected_cost = float(probabilities @ route_costs)
    p_opt = float(probabilities[optimal_mask].sum())
    bsp = float(probabilities[marked].sum())

    if not (float(route_costs.min()) - 1e-9 <= expected_cost <= float(route_costs.max()) + 1e-9):
        raise RuntimeError("grover_cost_phase_expected_cost_out_of_range")
    if not (0.0 <= p_opt <= 1.0 + 1e-9):
        raise RuntimeError("grover_cost_phase_p_opt_out_of_range")

    return {
        "p_feas": 1.0,
        "p_opt": p_opt,
        "invalid_mass": 0.0,
        "expected_cost_full": expected_cost,
        "expected_cost_feasible": expected_cost,
        "feasible_conditional_p_opt": p_opt,
        "bsp": bsp,
        "probabilities": probabilities,
    }


def run_cost_phase_grover_chain(
    context: StudyContext,
    *,
    depths: Sequence[int],
    seed: int,
) -> list[DepthRow]:
    """Optimize the cost-phase Grover-Mixer with layerwise continuation."""

    requested_depths = [int(depth) for depth in depths]
    if not requested_depths or requested_depths != sorted(set(requested_depths)):
        raise ValueError("depths_must_be_nonempty_sorted_unique")

    route_cost_phase, marked, _ = cost_phase_setup(context)
    rows: list[DepthRow] = []
    previous_optimum: np.ndarray | None = None

    for depth in requested_depths:
        x0 = continuation_initial_parameters(
            previous_optimum,
            depth=depth,
            seed=seed,
        )
        budget = eval_budget_for_depth(depth)

        def objective(parameters: np.ndarray) -> float:
            diagnostics = evaluate_cost_phase_grover(
                context,
                parameters,
                depth,
                route_cost_phase=route_cost_phase,
                marked=marked,
            )
            return float(diagnostics["expected_cost_full"])

        outcome = cobyla_with_explicit_x0(
            objective,
            x0,
            depth=depth,
            evaluation_budget=budget,
        )
        diagnostics = evaluate_cost_phase_grover(
            context,
            np.asarray(outcome.parameters, dtype=np.float64),
            depth,
            route_cost_phase=route_cost_phase,
            marked=marked,
        )
        exact_cost = float(context.exact_cost)
        regret = (
            float(diagnostics["expected_cost_full"]) - exact_cost
        ) / max(exact_cost, 1e-12)

        row = DepthRow(
            algorithm=GROVER_COST_PHASE,
            depth=depth,
            seed=int(seed),
            representation="logical_feasible_routes_20_routes",
            search_dimension=int(context.feasible_metadata.dimension),
            mixer_type="H_G,F=|s_F><s_F| (rank-one feasible Grover mixer)",
            parameter_count=2 * depth,
            initialization_source=(
                "depth_derived_seed" if previous_optimum is None
                else f"continuation_from_p{requested_depths[requested_depths.index(depth) - 1]}_plus_seed_small_angle"
            ),
            evaluation_budget=budget,
            evaluation_budget_used=outcome.evaluations,
            optimizer_reason=outcome.reason,
            optimizer_success=outcome.success,
            optimizer_wall_time_seconds=outcome.wall_time,
            p_feas=float(diagnostics["p_feas"]),
            p_opt=float(diagnostics["p_opt"]),
            invalid_mass=float(diagnostics["invalid_mass"]),
            expected_cost_full=float(diagnostics["expected_cost_full"]),
            expected_cost_feasible=float(diagnostics["expected_cost_feasible"]),
            feasible_conditional_p_opt=float(diagnostics["feasible_conditional_p_opt"]),
            normalized_regret=float(regret),
            bsp=float(diagnostics["bsp"]),
            parent_depth_parameters=(
                None
                if previous_optimum is None
                else tuple(map(float, previous_optimum))
            ),
            final_parameters=tuple(outcome.parameters),
        )
        rows.append(row)
        previous_optimum = np.asarray(outcome.parameters, dtype=np.float64)

        print(
            f"  [grover_cost_phase] p={depth:3d}  "
            f"p_opt={row.p_opt:.6f}  p_feas={row.p_feas:.6f}  "
            f"E[C]={row.expected_cost_full:.6f}  "
            f"evals={outcome.evaluations:3d}/{budget}  "
            f"t={outcome.wall_time:6.2f}s  {outcome.reason}"
        )

    return rows


def budget_limited_tags(rows: Sequence[DepthRow]) -> list[dict[str, str]]:
    """Classify optimization outcomes without calling evaluation caps timeouts."""

    tags: list[dict[str, str]] = []
    previous_by_algorithm: dict[str, DepthRow] = {}
    for row in rows:
        status = "COMPLETED"
        if row.optimizer_reason.startswith("evaluation_budget_exhausted"):
            status = "BUDGET_LIMITED"
        elif row.optimizer_reason.startswith("optimizer_failed"):
            status = "OPTIMIZER_FAILED"

        previous = previous_by_algorithm.get(row.algorithm)
        if status == "COMPLETED" and previous is not None:
            delta = row.p_opt - previous.p_opt
            if delta < -0.005:
                status = "REGRESSION"
            elif abs(delta) < 1e-9:
                status = "NO_IMPROVEMENT"

        tags.append(
            {
                "algorithm": row.algorithm,
                "depth": str(row.depth),
                "status": status,
                "optimizer_reason": row.optimizer_reason,
            }
        )
        previous_by_algorithm[row.algorithm] = row
    return tags


def save_cost_phase_distributions(
    context: StudyContext,
    rows: Sequence[DepthRow],
    path: Path,
) -> None:
    route_cost_phase, marked, _ = cost_phase_setup(context)
    payload: dict[str, np.ndarray] = {}
    for row in rows:
        diagnostics = evaluate_cost_phase_grover(
            context,
            np.asarray(row.final_parameters, dtype=np.float64),
            row.depth,
            route_cost_phase=route_cost_phase,
            marked=marked,
        )
        key = f"{row.algorithm}_p{row.depth}_seed{row.seed}"
        payload[key] = np.asarray(diagnostics["probabilities"], dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def write_cost_phase_run(
    *,
    depths: Sequence[int],
    seed: int,
    penalty: float,
    result_root: Path,
) -> dict[str, Any]:
    """Run and save the cost-phase Grover trajectory only."""

    context = prepare_study_context(penalty=float(penalty))
    rows = run_cost_phase_grover_chain(context, depths=depths, seed=seed)

    result_root = Path(result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    write_canonical_rows(rows, result_root / "canonical_rows.csv")
    write_parameters(rows, result_root / "optimized_parameters.json")
    write_runtime_profile(rows, result_root / "runtime_profile.csv")
    (result_root / "failure_taxonomy.json").write_text(
        json.dumps(budget_limited_tags(rows), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    save_cost_phase_distributions(
        context,
        rows,
        result_root / "final_distributions.npz",
    )

    route_cost_phase, marked, _ = cost_phase_setup(context)
    metadata = {
        "algorithm": GROVER_COST_PHASE,
        "phase_operator": "route_cost_diagonal",
        "phase_formula": "exp(-i * gamma * C(P_i))",
        "route_costs": route_cost_phase.tolist(),
        "route_cost_unique_values": sorted(map(float, np.unique(route_cost_phase))),
        "incumbent_threshold_used_in_evolution": False,
        "threshold_mask_used_only_for_bsp": True,
        "marked_route_count": int(marked.sum()),
        "depths": list(map(int, depths)),
        "base_seed": int(seed),
        "depth_rng_seed_formula": "base_seed + 7919 * p",
        "penalty": float(penalty),
        "optimizer": "COBYLA",
        "evaluation_budget_formula": (
            f"B(p)={EVAL_BUDGET_BASE}+{EVAL_BUDGET_PER_PARAM}*2p, "
            f"clamped to [{EVAL_BUDGET_MIN},{EVAL_BUDGET_MAX}]"
        ),
        "gamma_bounds": list(GAMMA_BOUNDS),
        "beta_bounds": list(BETA_BOUNDS),
        "new_layer_gamma_scale": NEW_LAYER_GAMMA_SCALE,
        "new_layer_beta_scale": NEW_LAYER_BETA_SCALE,
    }
    (result_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "result_root": str(result_root),
        "algorithm": GROVER_COST_PHASE,
        "row_count": len(rows),
        "depth_range": [min(map(int, depths)), max(map(int, depths))],
        "budget_limited_rows": sum(
            row.optimizer_reason.startswith("evaluation_budget_exhausted")
            for row in rows
        ),
        "total_wall_time_seconds": sum(
            row.optimizer_wall_time_seconds for row in rows
        ),
    }

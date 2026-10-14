"""Stage 5 -- the actual solver wrapper, and the single most important
discipline in this whole engine: NEVER trust ``scipy.optimize``'s own
``success`` flag as the real definition of convergence. Every solve
here is followed by an independent, real post-solve check --
``constraints.check_all`` against the actual returned weights, plus
``is_risk_parity_converged`` when the objective is risk parity -- and
``OptimizationResult.converged`` reflects THAT, not the solver's own
report. This is not a hypothetical precaution: a risk-parity optimizer
in this account's own systematic-equity-research repo once reported
``success: True`` while leaving weights at the unconverged starting
guess, caught only by directly decomposing real risk contributions
after the fact.

Scipy constraint translation is deliberately simple and explicit
(per-type, not a generic framework) for the five concrete constraint
types Stage 3 defines -- a real, working v1, not a general-purpose
constrained-optimization framework.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from ..config import OptimizerConfig
from ..constraints import (
    FactorExposureConstraint,
    GrossLeverageConstraint,
    MaxPositionWeightConstraint,
    SectorExposureConstraint,
    TurnoverConstraint,
    check_all,
)
from ..data.schemas import OptimizationInput, OptimizationResult
from .objectives import (
    array_to_weights,
    is_risk_parity_converged,
    mean_variance_objective,
    min_variance_objective,
    risk_parity_objective,
    weights_to_array,
)

_OBJECTIVES = {"mean_variance", "min_variance", "risk_parity"}


def _position_bounds(constraints: Sequence, n: int) -> list[tuple[float, float]]:
    position_constraints = [c for c in constraints if isinstance(c, MaxPositionWeightConstraint)]
    if not position_constraints:
        return [(-1.0, 1.0)] * n  # a real, permissive default -- never unbounded
    # the tightest real bound wins when more than one is supplied
    lo = max((0.0 if c.long_only else -c.max_weight) for c in position_constraints)
    hi = min(c.max_weight for c in position_constraints)
    return [(lo, hi)] * n


def _scipy_inequality_constraints(constraints: Sequence, opt_input: OptimizationInput, security_ids: Sequence[str]) -> list[dict]:
    """Each non-position constraint becomes one real scipy ``'ineq'``
    constraint dict, ``fun(w) >= 0`` meaning satisfied. ``MaxPositionWeightConstraint``
    is handled separately, via real per-variable ``bounds`` -- a tighter,
    more efficient real formulation than an inequality constraint for a
    simple box bound.
    """
    scipy_constraints: list[dict] = []
    for constraint in constraints:
        if isinstance(constraint, SectorExposureConstraint):
            def fun(w, c=constraint):
                sector_totals: dict[str, float] = {}
                for i, sid in enumerate(security_ids):
                    sector = c.sector_by_security.get(sid, "unknown")
                    sector_totals[sector] = sector_totals.get(sector, 0.0) + w[i]
                return np.array([c.max_abs_sector_weight - abs(total) for total in sector_totals.values()])
            scipy_constraints.append({"type": "ineq", "fun": fun})
        elif isinstance(constraint, FactorExposureConstraint):
            def fun(w, c=constraint):
                values = []
                for factor_name, limit in c.max_abs_exposure.items():
                    if factor_name not in c.exposure_matrix.columns:
                        continue
                    column = c.exposure_matrix[factor_name]
                    net_exposure = sum(w[i] * column.get(sid, 0.0) for i, sid in enumerate(security_ids))
                    values.append(limit - abs(net_exposure))
                return np.array(values) if values else np.array([0.0])
            scipy_constraints.append({"type": "ineq", "fun": fun})
        elif isinstance(constraint, TurnoverConstraint):
            def fun(w, c=constraint):
                current = np.array([opt_input.current_weight(sid) for sid in security_ids])
                turnover = float(np.sum(np.abs(w - current)))
                return np.array([c.max_turnover - turnover])
            scipy_constraints.append({"type": "ineq", "fun": fun})
        elif isinstance(constraint, GrossLeverageConstraint):
            def fun(w, c=constraint):
                return np.array([c.max_gross_leverage - float(np.sum(np.abs(w)))])
            scipy_constraints.append({"type": "ineq", "fun": fun})
    return scipy_constraints


def optimize_portfolio(
    opt_input: OptimizationInput,
    objective: str,
    constraints: Sequence,
    optimizer_cfg: OptimizerConfig,
    portfolio_id: str = "portfolio",
    cost_model=None,
) -> OptimizationResult:
    """Solve, then independently verify -- the real two-step discipline
    this whole module exists to enforce. ``objective`` is one of
    ``"mean_variance"``, ``"min_variance"``, ``"risk_parity"``.

    ``cost_model`` (Stage 7, optional, defaults to ``None`` -- fully
    backward compatible with every earlier call site): a real
    ``TransactionCostModel``. When supplied with ``objective=
    "mean_variance"``, its estimated cost is subtracted from the
    optimizer's own utility, so the optimizer genuinely trades off
    expected alpha against the real cost of getting there -- not
    applied only after the fact. For ``min_variance``/``risk_parity``,
    the cost model is used only to report the real estimated
    ``transaction_cost`` on the result; those objectives don't have an
    expected-return term for a cost to trade off against.
    """
    if objective not in _OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}, must be one of {sorted(_OBJECTIVES)}")

    security_ids = list(opt_input.security_ids)
    n = len(security_ids)
    expected_returns = weights_to_array(opt_input.expected_returns, security_ids)
    covariance = opt_input.covariance.loc[security_ids, security_ids].to_numpy()

    if objective == "mean_variance":
        def scipy_objective(w):
            utility = mean_variance_objective(w, expected_returns, covariance, optimizer_cfg.risk_aversion)
            if cost_model is not None:
                utility += cost_model.estimate_cost(array_to_weights(w, security_ids), opt_input.current_weights, security_ids)
            return utility
    elif objective == "min_variance":
        def scipy_objective(w):
            return min_variance_objective(w, covariance)
    else:
        def scipy_objective(w):
            return risk_parity_objective(w, covariance)

    bounds = _position_bounds(constraints, n)
    scipy_constraints = _scipy_inequality_constraints(constraints, opt_input, security_ids)
    scipy_constraints.append({"type": "eq", "fun": lambda w: float(np.sum(w)) - optimizer_cfg.target_net_exposure})

    initial_guess = np.full(n, optimizer_cfg.target_net_exposure / n)
    solver_result = minimize(
        scipy_objective, initial_guess, method="SLSQP", bounds=bounds, constraints=scipy_constraints,
        options={"maxiter": optimizer_cfg.max_iterations, "ftol": 1e-12},
    )

    target_weights = array_to_weights(solver_result.x, security_ids)

    warnings: list[str] = []
    violations = check_all(constraints, target_weights, opt_input, tolerance=optimizer_cfg.constraint_tolerance)
    net_exposure = float(np.sum(solver_result.x))
    net_exposure_ok = abs(net_exposure - optimizer_cfg.target_net_exposure) <= optimizer_cfg.constraint_tolerance
    if not net_exposure_ok:
        warnings.append(f"net exposure {net_exposure:.6f} does not match target {optimizer_cfg.target_net_exposure}")

    risk_parity_ok = True
    if objective == "risk_parity":
        risk_parity_ok = is_risk_parity_converged(solver_result.x, covariance, optimizer_cfg.risk_parity_tolerance)
        if not risk_parity_ok:
            warnings.append("risk contributions are not genuinely equal within the configured tolerance")

    converged = not violations and net_exposure_ok and risk_parity_ok
    if not solver_result.success and converged:
        # the solver itself reported failure, but our OWN independent
        # check says the real solution is feasible and (if applicable)
        # genuinely risk-balanced -- trust the real check, not the flag,
        # but note the disagreement for a researcher to see
        warnings.append(f"scipy reported success=False ({solver_result.message}) but independent verification found the solution feasible")
    if solver_result.success and not converged:
        warnings.append(f"scipy reported success=True but independent verification found real violations -- the exact failure mode this module exists to catch")
    for violation in violations:
        warnings.append(f"{violation.constraint_name}: {violation.detail}")

    expected_return = float(solver_result.x @ expected_returns)
    expected_variance = float(solver_result.x @ covariance @ solver_result.x)
    turnover = float(np.sum(np.abs(solver_result.x - np.array([opt_input.current_weight(sid) for sid in security_ids]))))
    transaction_cost = (
        cost_model.estimate_cost(target_weights, opt_input.current_weights, security_ids) if cost_model is not None else 0.0
    )

    return OptimizationResult(
        as_of=opt_input.as_of,
        portfolio_id=portfolio_id,
        target_weights=target_weights,
        expected_return=expected_return,
        expected_variance=expected_variance,
        turnover=turnover,
        converged=converged,
        transaction_cost=transaction_cost,
        # a real, honest simplification: when NOT converged, these are
        # the VIOLATED constraint names (for triage), not genuinely
        # "active but satisfied" ones -- detecting a truly binding (at
        # its limit, still feasible) constraint needs a real margin
        # check this v1 doesn't yet do, left for a later refinement
        # rather than claimed now
        binding_constraints=tuple(sorted({v.constraint_name for v in violations})),
        warnings=tuple(warnings),
    )

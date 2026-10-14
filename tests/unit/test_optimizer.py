import numpy as np
import pandas as pd
import pytest

from portfolio_construction.config import OptimizerConfig
from portfolio_construction.constraints import GrossLeverageConstraint, MaxPositionWeightConstraint, TurnoverConstraint
from portfolio_construction.data import OptimizationInput
from portfolio_construction.optimization import optimize_portfolio


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _opt_input(security_ids, expected_returns, covariance, current_weights=None) -> OptimizationInput:
    return OptimizationInput(
        as_of=_ts("2020-06-30"), security_ids=tuple(security_ids), expected_returns=expected_returns,
        covariance=covariance, current_weights=current_weights or {},
    )


def test_min_variance_optimizer_converges_and_respects_bounds():
    sids = ["A", "B", "C"]
    cov = pd.DataFrame(np.diag([0.01, 0.04, 0.09]), index=sids, columns=sids)
    opt_input = _opt_input(sids, {s: 0.0 for s in sids}, cov)
    constraints = [MaxPositionWeightConstraint(max_weight=0.8, long_only=True)]

    result = optimize_portfolio(opt_input, "min_variance", constraints, OptimizerConfig())
    assert result.converged
    assert result.warnings == ()
    assert sum(result.target_weights.values()) == pytest.approx(1.0, abs=1e-6)
    # min-variance should tilt heavily toward the lowest-variance asset
    assert result.target_weights["A"] > result.target_weights["C"]


def test_risk_parity_optimizer_produces_genuinely_equal_risk_contributions():
    sids = ["A", "B", "C"]
    cov = pd.DataFrame(np.diag([0.01, 0.04, 0.09]), index=sids, columns=sids)
    opt_input = _opt_input(sids, {s: 0.0 for s in sids}, cov)

    result = optimize_portfolio(opt_input, "risk_parity", [], OptimizerConfig())
    assert result.converged
    w = np.array([result.target_weights[s] for s in sids])
    contributions = w * (cov.to_numpy() @ w)
    relative_spread = np.max(np.abs(contributions / contributions.mean() - 1.0))
    assert relative_spread < 0.05


def test_mean_variance_optimizer_respects_a_real_turnover_constraint():
    sids = ["A", "B"]
    cov = pd.DataFrame(np.eye(2) * 0.02, index=sids, columns=sids)
    opt_input = _opt_input(sids, {"A": 0.1, "B": 0.0}, cov, current_weights={"A": 0.5, "B": 0.5})
    constraints = [MaxPositionWeightConstraint(max_weight=1.0, long_only=True), TurnoverConstraint(max_turnover=0.1)]

    result = optimize_portfolio(opt_input, "mean_variance", constraints, OptimizerConfig())
    assert result.converged
    assert result.turnover <= 0.1 + 1e-4


def test_optimizer_rejects_an_unknown_objective():
    sids = ["A"]
    cov = pd.DataFrame([[0.02]], index=sids, columns=sids)
    opt_input = _opt_input(sids, {"A": 0.01}, cov)
    with pytest.raises(ValueError):
        optimize_portfolio(opt_input, "nonexistent_objective", [], OptimizerConfig())


def test_optimizer_catches_an_infeasible_constraint_combination():
    # two position constraints that together make the equality
    # (fully-invested) constraint impossible to satisfy
    sids = ["A", "B"]
    cov = pd.DataFrame(np.eye(2) * 0.02, index=sids, columns=sids)
    opt_input = _opt_input(sids, {"A": 0.01, "B": 0.01}, cov)
    constraints = [MaxPositionWeightConstraint(max_weight=0.3, long_only=True)]  # max gross = 0.6 < target 1.0

    result = optimize_portfolio(opt_input, "min_variance", constraints, OptimizerConfig())
    assert not result.converged
    assert len(result.warnings) > 0


def test_optimizer_respects_gross_leverage_for_a_long_short_book():
    sids = ["A", "B"]
    cov = pd.DataFrame(np.eye(2) * 0.02, index=sids, columns=sids)
    opt_input = _opt_input(sids, {"A": 0.05, "B": -0.05}, cov)
    constraints = [MaxPositionWeightConstraint(max_weight=1.0, long_only=False), GrossLeverageConstraint(max_gross_leverage=1.0)]
    cfg = OptimizerConfig(target_net_exposure=0.0)  # dollar-neutral long/short

    result = optimize_portfolio(opt_input, "mean_variance", constraints, cfg)
    assert result.converged
    gross = sum(abs(w) for w in result.target_weights.values())
    assert gross <= 1.0 + 1e-4

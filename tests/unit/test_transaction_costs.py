import numpy as np
import pandas as pd
import pytest

from portfolio_construction.backtest import TransactionCostModel
from portfolio_construction.config import OptimizerConfig
from portfolio_construction.constraints import MaxPositionWeightConstraint
from portfolio_construction.data import OptimizationInput
from portfolio_construction.optimization import optimize_portfolio


def test_estimate_cost_matches_hand_computed_linear_cost():
    model = TransactionCostModel(linear_cost_bps=20.0)
    cost = model.estimate_cost({"A": 0.6, "B": 0.4}, {"A": 0.5, "B": 0.5}, ["A", "B"])
    # trades: |0.6-0.5| + |0.4-0.5| = 0.2, cost = 0.2 * 20/10000
    assert cost == pytest.approx(0.2 * 20.0 / 10_000.0)


def test_estimate_cost_treats_missing_weights_as_a_real_zero():
    model = TransactionCostModel(linear_cost_bps=10.0)
    cost = model.estimate_cost({"A": 0.5}, {}, ["A"])  # a brand new position from nothing
    assert cost == pytest.approx(0.5 * 10.0 / 10_000.0)


def test_estimate_cost_adds_a_real_market_impact_term():
    model = TransactionCostModel(linear_cost_bps=0.0, market_impact_coefficient=0.01)
    cost = model.estimate_cost({"A": 0.5}, {"A": 0.0}, ["A"])
    assert cost == pytest.approx(0.01 * 0.5**1.5)


def test_transaction_cost_model_rejects_negative_parameters():
    with pytest.raises(ValueError):
        TransactionCostModel(linear_cost_bps=-1.0)
    with pytest.raises(ValueError):
        TransactionCostModel(market_impact_coefficient=-1.0)


def test_optimizer_reports_a_real_transaction_cost_when_a_cost_model_is_supplied():
    sids = ["A", "B"]
    cov = pd.DataFrame(np.eye(2) * 0.02, index=sids, columns=sids)
    opt_input = OptimizationInput(
        as_of=pd.Timestamp("2020-06-30"), security_ids=tuple(sids), expected_returns={"A": 0.0, "B": 0.0},
        covariance=cov, current_weights={"A": 0.5, "B": 0.5},
    )
    result = optimize_portfolio(opt_input, "min_variance", [], OptimizerConfig(), cost_model=TransactionCostModel(linear_cost_bps=50.0))
    assert result.transaction_cost >= 0.0
    assert result.transaction_cost == pytest.approx(result.turnover * 50.0 / 10_000.0)


def test_optimizer_without_a_cost_model_reports_zero_cost():
    sids = ["A"]
    cov = pd.DataFrame([[0.02]], index=sids, columns=sids)
    opt_input = OptimizationInput(as_of=pd.Timestamp("2020-06-30"), security_ids=tuple(sids), expected_returns={"A": 0.01}, covariance=cov)
    result = optimize_portfolio(opt_input, "min_variance", [], OptimizerConfig())
    assert result.transaction_cost == 0.0


def test_cost_aware_mean_variance_optimizer_trades_less_when_costs_are_high():
    sids = ["A", "B"]
    cov = pd.DataFrame(np.eye(2) * 0.02, index=sids, columns=sids)
    # a real alpha edge favoring B, starting from an all-A book
    opt_input = OptimizationInput(
        as_of=pd.Timestamp("2020-06-30"), security_ids=tuple(sids), expected_returns={"A": 0.0, "B": 0.05},
        covariance=cov, current_weights={"A": 1.0, "B": 0.0},
    )
    constraints = [MaxPositionWeightConstraint(max_weight=1.0, long_only=True)]
    cfg = OptimizerConfig(risk_aversion=1.0)

    no_cost = optimize_portfolio(opt_input, "mean_variance", constraints, cfg)
    high_cost = optimize_portfolio(opt_input, "mean_variance", constraints, cfg, cost_model=TransactionCostModel(linear_cost_bps=5000.0))
    assert high_cost.turnover < no_cost.turnover

"""End-to-end: a real quarterly rebalancing schedule, backtested net of
real transaction costs, entirely on the synthetic dataset.
"""

import pandas as pd
import pytest
from factor_model.config import EngineConfig as FactorModelConfig, UniverseConfig
from factor_model.data import SyntheticDataConfig, fundamental_observations_from_frame, generate_synthetic_dataset, market_observations_from_frame

from portfolio_construction.backtest import TransactionCostModel, run_backtest
from portfolio_construction.config import EngineConfig
from portfolio_construction.constraints import GrossLeverageConstraint, MaxPositionWeightConstraint
from portfolio_construction.rebalancing import run_rebalancing_schedule


def _zero_expected_returns(as_of, security_ids):
    return {sid: 0.0 for sid in security_ids}


def _standard_constraints(as_of, exposure_matrix):
    return [MaxPositionWeightConstraint(max_weight=0.2, long_only=True), GrossLeverageConstraint(max_gross_leverage=1.0)]


def test_backtest_of_a_real_rebalancing_schedule_on_the_synthetic_dataset():
    cfg = SyntheticDataConfig(n_securities=25, n_sectors=5, n_periods_months=36)
    dataset = generate_synthetic_dataset(cfg)
    market_obs = market_observations_from_frame(dataset.market)
    fundamental_obs = fundamental_observations_from_frame(dataset.fundamentals)
    dates = sorted(dataset.market["as_of"].unique())
    rebalance_dates = [pd.Timestamp(d) for d in dates[12::3]]

    fm_cfg = FactorModelConfig(universe=UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0))
    pc_cfg = EngineConfig()

    schedule = run_rebalancing_schedule(
        market_obs, fundamental_obs, rebalance_dates, lookback_periods=12, objective="min_variance",
        expected_returns_source=_zero_expected_returns, constraints_factory=_standard_constraints,
        fm_engine_cfg=fm_cfg, pc_engine_cfg=pc_cfg,
    )
    assert all(result.converged for result in schedule.results)

    backtest = run_backtest(schedule.results, market_obs)
    assert len(backtest.net_returns) == len(schedule.results) - 1
    assert backtest.cumulative_net_return == pytest.approx(float((1.0 + backtest.net_returns).prod() - 1.0))


def test_backtest_with_real_transaction_costs_nets_below_the_gross_schedule():
    cfg = SyntheticDataConfig(n_securities=25, n_sectors=5, n_periods_months=36)
    dataset = generate_synthetic_dataset(cfg)
    market_obs = market_observations_from_frame(dataset.market)
    fundamental_obs = fundamental_observations_from_frame(dataset.fundamentals)
    dates = sorted(dataset.market["as_of"].unique())
    rebalance_dates = [pd.Timestamp(d) for d in dates[12::3]]

    fm_cfg = FactorModelConfig(universe=UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0))
    pc_cfg = EngineConfig()

    no_cost_schedule = run_rebalancing_schedule(
        market_obs, fundamental_obs, rebalance_dates, lookback_periods=12, objective="min_variance",
        expected_returns_source=_zero_expected_returns, constraints_factory=_standard_constraints,
        fm_engine_cfg=fm_cfg, pc_engine_cfg=pc_cfg,
    )
    costed_schedule = run_rebalancing_schedule(
        market_obs, fundamental_obs, rebalance_dates, lookback_periods=12, objective="min_variance",
        expected_returns_source=_zero_expected_returns, constraints_factory=_standard_constraints,
        fm_engine_cfg=fm_cfg, pc_engine_cfg=pc_cfg, cost_model=TransactionCostModel(linear_cost_bps=50.0),
    )

    backtest_no_cost = run_backtest(no_cost_schedule.results, market_obs)
    backtest_with_cost = run_backtest(costed_schedule.results, market_obs)
    assert backtest_with_cost.cumulative_net_return < backtest_no_cost.cumulative_net_return

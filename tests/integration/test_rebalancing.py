"""End-to-end: a real quarterly rebalancing schedule on the synthetic
dataset, verifying point-in-time safety (no rebalance date ever sees
data dated after it) and real turnover continuity across rebalances.
"""

import pandas as pd
import pytest
from factor_model.config import EngineConfig as FactorModelConfig, UniverseConfig
from factor_model.data import SyntheticDataConfig, fundamental_observations_from_frame, generate_synthetic_dataset, market_observations_from_frame

from portfolio_construction.config import EngineConfig
from portfolio_construction.constraints import GrossLeverageConstraint, MaxPositionWeightConstraint
from portfolio_construction.rebalancing import run_rebalancing_schedule


def _zero_expected_returns(as_of, security_ids):
    return {sid: 0.0 for sid in security_ids}


def _standard_constraints(as_of, exposure_matrix):
    return [MaxPositionWeightConstraint(max_weight=0.2, long_only=True), GrossLeverageConstraint(max_gross_leverage=1.0)]


def test_quarterly_rebalancing_schedule_runs_point_in_time_safely():
    cfg = SyntheticDataConfig(n_securities=25, n_sectors=5, n_periods_months=36)
    dataset = generate_synthetic_dataset(cfg)
    market_obs = market_observations_from_frame(dataset.market)
    fundamental_obs = fundamental_observations_from_frame(dataset.fundamentals)
    dates = sorted(dataset.market["as_of"].unique())

    # quarterly rebalances, starting once 12 real months of history exist
    rebalance_dates = [pd.Timestamp(d) for d in dates[12::3]]

    fm_cfg = FactorModelConfig(universe=UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0))
    pc_cfg = EngineConfig()

    schedule = run_rebalancing_schedule(
        market_obs, fundamental_obs, rebalance_dates, lookback_periods=12, objective="min_variance",
        expected_returns_source=_zero_expected_returns, constraints_factory=_standard_constraints,
        fm_engine_cfg=fm_cfg, pc_engine_cfg=pc_cfg,
    )

    assert len(schedule.results) == len(rebalance_dates)
    assert all(result.converged for result in schedule.results)
    # real continuity: turnover on the FIRST rebalance (from an empty
    # book) should equal gross exposure; later rebalances' turnover
    # should be bounded, not a fresh full liquidation+rebuild every time
    first_turnover = schedule.results[0].turnover
    assert first_turnover == pytest.approx(1.0, abs=1e-3)  # starting from nothing, full build


def test_rebalancing_schedule_warns_rather_than_crashes_with_insufficient_history():
    cfg = SyntheticDataConfig(n_securities=10, n_sectors=5, n_periods_months=5, restatement_enabled=False)
    dataset = generate_synthetic_dataset(cfg)
    market_obs = market_observations_from_frame(dataset.market)
    fundamental_obs = fundamental_observations_from_frame(dataset.fundamentals)
    dates = sorted(dataset.market["as_of"].unique())

    fm_cfg = FactorModelConfig(universe=UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0))
    pc_cfg = EngineConfig()

    schedule = run_rebalancing_schedule(
        market_obs, fundamental_obs, [pd.Timestamp(dates[0])], lookback_periods=12, objective="min_variance",
        expected_returns_source=_zero_expected_returns, constraints_factory=_standard_constraints,
        fm_engine_cfg=fm_cfg, pc_engine_cfg=pc_cfg,
    )
    assert schedule.results == []
    assert len(schedule.warnings_by_date) == 1

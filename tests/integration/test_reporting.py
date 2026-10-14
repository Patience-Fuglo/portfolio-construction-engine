import json

import pandas as pd
import pytest
from factor_model.config import EngineConfig as FactorModelConfig, UniverseConfig
from factor_model.data import SyntheticDataConfig, fundamental_observations_from_frame, generate_synthetic_dataset, market_observations_from_frame

from portfolio_construction.backtest import run_backtest
from portfolio_construction.config import EngineConfig
from portfolio_construction.constraints import GrossLeverageConstraint, MaxPositionWeightConstraint
from portfolio_construction.rebalancing import run_rebalancing_schedule
from portfolio_construction.reporting import render_markdown_report, report_to_dict


def _zero_expected_returns(as_of, security_ids):
    return {sid: 0.0 for sid in security_ids}


def _standard_constraints(as_of, exposure_matrix):
    return [MaxPositionWeightConstraint(max_weight=0.2, long_only=True), GrossLeverageConstraint(max_gross_leverage=1.0)]


@pytest.fixture(scope="module")
def schedule_and_backtest():
    cfg = SyntheticDataConfig(n_securities=20, n_sectors=5, n_periods_months=30)
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
    backtest = run_backtest(schedule.results, market_obs)
    return schedule, backtest


def test_report_to_dict_is_json_serializable(schedule_and_backtest):
    schedule, backtest = schedule_and_backtest
    payload = report_to_dict(schedule, backtest, "benchmark_min_variance")
    json.dumps(payload)


def test_report_to_dict_reconciles_with_the_real_underlying_objects(schedule_and_backtest):
    schedule, backtest = schedule_and_backtest
    payload = report_to_dict(schedule, backtest, "benchmark_min_variance")
    assert payload["n_rebalances"] == len(schedule.results)
    assert payload["all_converged"] == all(r.converged for r in schedule.results)
    assert payload["backtest"]["cumulative_net_return"] == pytest.approx(backtest.cumulative_net_return)


def test_render_markdown_report_contains_every_real_section(schedule_and_backtest):
    schedule, backtest = schedule_and_backtest
    markdown = render_markdown_report(schedule, backtest, "benchmark_min_variance")
    for heading in ["## Rebalancing Schedule", "## Final Target Weights", "## Backtest"]:
        assert heading in markdown
    assert "benchmark_min_variance" in markdown
    assert "synthetic" in markdown.lower()


def test_report_rendering_is_deterministic(schedule_and_backtest):
    schedule, backtest = schedule_and_backtest
    first = render_markdown_report(schedule, backtest, "P")
    second = render_markdown_report(schedule, backtest, "P")
    assert first == second
    assert report_to_dict(schedule, backtest, "P") == report_to_dict(schedule, backtest, "P")

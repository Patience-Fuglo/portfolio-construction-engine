"""End-to-end: optimize a real portfolio using a stock covariance
matrix built directly from factor-model-engine's own real pipeline
output, not a hand-built fixture.
"""

import pandas as pd
import pytest
from factor_model.config import EngineConfig as FactorModelConfig, UniverseConfig
from factor_model.data import SyntheticDataConfig, build_point_in_time_universe, fundamental_observations_from_frame, generate_synthetic_dataset, market_observations_from_frame
from factor_model.factors import compute_factor_exposure_table
from factor_model.regression import compute_factor_returns
from factor_model.risk import compute_factor_covariance, compute_specific_returns, compute_specific_variance

from portfolio_construction.config import OptimizerConfig, RiskModelConfig
from portfolio_construction.constraints import GrossLeverageConstraint, MaxPositionWeightConstraint
from portfolio_construction.data import OptimizationInput
from portfolio_construction.optimization import optimize_portfolio
from portfolio_construction.risk_model import build_exposure_matrix, build_stock_covariance_matrix


def test_min_variance_optimization_on_a_real_factor_model_engine_risk_model():
    cfg = SyntheticDataConfig(n_securities=25, n_sectors=5, n_periods_months=36)
    dataset = generate_synthetic_dataset(cfg)
    market_obs = market_observations_from_frame(dataset.market)
    fundamental_obs = fundamental_observations_from_frame(dataset.fundamentals)
    dates = sorted(dataset.market["as_of"].unique())
    period_dates = [pd.Timestamp(d) for d in dates[12:]]
    as_of = period_dates[-1]

    fm_cfg = FactorModelConfig(universe=UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0))
    universe = build_point_in_time_universe(market_obs, as_of, fm_cfg.universe, fm_cfg.data)
    stock_exposure_rows, _ = compute_factor_exposure_table(market_obs, fundamental_obs, as_of, universe, fm_cfg)
    return_rows, _ = compute_factor_returns(market_obs, fundamental_obs, period_dates, fm_cfg)
    factor_cov = compute_factor_covariance(return_rows, fm_cfg.risk)
    specific_returns = compute_specific_returns(market_obs, fundamental_obs, period_dates, fm_cfg)
    specific_variance = compute_specific_variance(specific_returns)

    exposure_matrix = build_exposure_matrix(stock_exposure_rows)
    covariance, _ = build_stock_covariance_matrix(exposure_matrix, factor_cov.covariance, specific_variance, RiskModelConfig())
    security_ids = covariance.index.tolist()

    opt_input = OptimizationInput(
        as_of=as_of, security_ids=tuple(security_ids),
        expected_returns={sid: 0.0 for sid in security_ids},  # min-variance doesn't use these
        covariance=covariance,
    )
    constraints = [MaxPositionWeightConstraint(max_weight=0.15, long_only=True), GrossLeverageConstraint(max_gross_leverage=1.0)]

    result = optimize_portfolio(opt_input, "min_variance", constraints, OptimizerConfig())
    assert result.converged
    assert sum(result.target_weights.values()) == pytest.approx(1.0, abs=1e-4)
    assert max(result.target_weights.values()) <= 0.15 + 1e-4
    assert all(w >= -1e-8 for w in result.target_weights.values())

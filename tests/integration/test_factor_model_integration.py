"""End-to-end: builds a real stock-level covariance matrix directly
from factor-model-engine's own real pipeline output on its synthetic
dataset -- a genuine cross-repo integration proof, not a hand-built
fixture standing in for the real dependency.
"""

import pandas as pd
from factor_model.config import DataConfig, EngineConfig as FactorModelConfig, UniverseConfig
from factor_model.data import build_point_in_time_universe, fundamental_observations_from_frame, generate_synthetic_dataset, market_observations_from_frame
from factor_model.data import SyntheticDataConfig
from factor_model.factors import compute_factor_exposure_table
from factor_model.regression import compute_factor_returns
from factor_model.risk import compute_factor_covariance, compute_specific_returns, compute_specific_variance

from portfolio_construction.config import RiskModelConfig
from portfolio_construction.risk_model import build_exposure_matrix, build_stock_covariance_matrix


def test_stock_covariance_from_the_real_factor_model_engine_pipeline():
    cfg = SyntheticDataConfig(n_securities=30, n_sectors=5, n_periods_months=36)
    dataset = generate_synthetic_dataset(cfg)
    market_obs = market_observations_from_frame(dataset.market)
    fundamental_obs = fundamental_observations_from_frame(dataset.fundamentals)
    dates = sorted(dataset.market["as_of"].unique())
    period_dates = [pd.Timestamp(d) for d in dates[12:]]
    as_of = period_dates[-1]

    fm_engine_cfg = FactorModelConfig(universe=UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0))
    universe = build_point_in_time_universe(market_obs, as_of, fm_engine_cfg.universe, fm_engine_cfg.data)
    stock_exposure_rows, _ = compute_factor_exposure_table(market_obs, fundamental_obs, as_of, universe, fm_engine_cfg)
    return_rows, _ = compute_factor_returns(market_obs, fundamental_obs, period_dates, fm_engine_cfg)
    factor_covariance_result = compute_factor_covariance(return_rows, fm_engine_cfg.risk)
    specific_returns = compute_specific_returns(market_obs, fundamental_obs, period_dates, fm_engine_cfg)
    specific_variance = compute_specific_variance(specific_returns)

    exposure_matrix = build_exposure_matrix(stock_exposure_rows)
    covariance, warnings = build_stock_covariance_matrix(
        exposure_matrix, factor_covariance_result.covariance, specific_variance, RiskModelConfig()
    )

    assert covariance.shape[0] == covariance.shape[1]
    assert set(covariance.index) == set(universe.included_ids())
    # a real stock covariance matrix: positive diagonal (every real
    # security has positive total variance) and symmetric
    assert (covariance.to_numpy().diagonal() > 0).all()
    import numpy as np

    np.testing.assert_allclose(covariance.to_numpy(), covariance.to_numpy().T)

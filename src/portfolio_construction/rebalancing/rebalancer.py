"""Stage 6 -- turnover-aware rebalancing across a real date range,
point-in-time safe: at EACH rebalance date, the risk model (factor
covariance, specific risk, exposures) is re-estimated using only
``lookback_periods`` of REAL trailing history ending at that date --
never a period dated after it. The previous rebalance's own target
weights become the next one's ``current_weights``, so turnover/
transaction-cost constraints see a real, continuous trading history,
not a fresh start every time.

Alpha (expected returns) is deliberately NOT generated here --
generating a real alpha signal is a different repo's job (this
account's `factor-model-engine` or a dedicated signal repo); this
engine takes expected returns as an explicit input per rebalance date,
consistent with its own real scope: construction, not signal research.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import pandas as pd
from factor_model.config import EngineConfig as FactorModelConfig
from factor_model.data.schemas import FundamentalObservation, MarketObservation
from factor_model.data.universe import build_point_in_time_universe
from factor_model.factors import compute_factor_exposure_table
from factor_model.regression import compute_factor_returns
from factor_model.risk import compute_factor_covariance, compute_specific_returns, compute_specific_variance

from ..config import EngineConfig
from ..data.schemas import OptimizationInput, OptimizationResult
from ..risk_model import build_exposure_matrix, build_stock_covariance_matrix

ExpectedReturnsSource = Callable[[pd.Timestamp, Sequence[str]], dict[str, float]]
ConstraintsFactory = Callable[[pd.Timestamp, pd.DataFrame], Sequence]


@dataclass
class RebalancingSchedule:
    results: list[OptimizationResult] = field(default_factory=list)
    warnings_by_date: dict[pd.Timestamp, list[str]] = field(default_factory=dict)


def run_rebalancing_schedule(
    market_observations: Sequence[MarketObservation],
    fundamental_observations: Sequence[FundamentalObservation],
    rebalance_dates: Sequence[pd.Timestamp],
    lookback_periods: int,
    objective: str,
    expected_returns_source: ExpectedReturnsSource,
    constraints_factory: ConstraintsFactory,
    fm_engine_cfg: FactorModelConfig,
    pc_engine_cfg: EngineConfig,
    portfolio_id: str = "portfolio",
    cost_model=None,
) -> RebalancingSchedule:
    """At each sorted ``rebalance_date``: build the real point-in-time
    universe and factor exposures AS OF that date, estimate the factor
    covariance/specific risk from the ``lookback_periods`` real monthly
    dates immediately preceding and including it (never a later one),
    build the stock covariance matrix, call the real optimizer, and
    carry its target weights forward as the NEXT rebalance's real
    pre-trade (current) weights. ``cost_model`` (Stage 7, optional) is
    passed straight through to ``optimize_portfolio`` at every
    rebalance date -- the same real transaction-cost model applies
    consistently across the whole schedule, not reconstructed per call.
    """
    schedule = RebalancingSchedule()
    all_market_dates = sorted({obs.as_of for obs in market_observations})
    current_weights: dict[str, float] = {}

    from ..optimization import optimize_portfolio  # local import avoids a circular import at module load time

    for rebalance_date in sorted(rebalance_dates):
        lookback_dates = [d for d in all_market_dates if d <= rebalance_date][-lookback_periods:]
        if len(lookback_dates) < 2:
            schedule.warnings_by_date[rebalance_date] = [f"skipped -- fewer than 2 real periods of history by {rebalance_date.date()}"]
            continue

        universe = build_point_in_time_universe(market_observations, rebalance_date, fm_engine_cfg.universe, fm_engine_cfg.data)
        stock_exposure_rows, _ = compute_factor_exposure_table(
            market_observations, fundamental_observations, rebalance_date, universe, fm_engine_cfg
        )
        return_rows, return_warnings = compute_factor_returns(market_observations, fundamental_observations, lookback_dates, fm_engine_cfg)
        factor_cov_result = compute_factor_covariance(return_rows, fm_engine_cfg.risk)
        specific_returns = compute_specific_returns(market_observations, fundamental_observations, lookback_dates, fm_engine_cfg)
        specific_variance = compute_specific_variance(specific_returns)

        exposure_matrix = build_exposure_matrix(stock_exposure_rows)
        covariance, covariance_warnings = build_stock_covariance_matrix(
            exposure_matrix, factor_cov_result.covariance, specific_variance, pc_engine_cfg.risk_model
        )
        security_ids = covariance.index.tolist()

        expected_returns = expected_returns_source(rebalance_date, security_ids)
        opt_input = OptimizationInput(
            as_of=rebalance_date, security_ids=tuple(security_ids), expected_returns=expected_returns,
            covariance=covariance, current_weights={sid: current_weights.get(sid, 0.0) for sid in security_ids},
        )
        constraints = constraints_factory(rebalance_date, exposure_matrix)

        result = optimize_portfolio(opt_input, objective, constraints, pc_engine_cfg.optimizer, portfolio_id, cost_model=cost_model)
        schedule.results.append(result)
        schedule.warnings_by_date[rebalance_date] = list(return_warnings) + list(factor_cov_result.warnings) + list(covariance_warnings)
        current_weights = result.target_weights

    return schedule

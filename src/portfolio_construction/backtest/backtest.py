"""Stage 8 -- forward-applies a real rebalancing schedule's target
weights and measures the REAL realized portfolio return over each
holding period, net of the real transaction cost already reported on
the rebalance that established those weights. Reuses
``factor-model-engine``'s own point-in-time realized-return helper
(never re-derives stock returns independently) and
``alpha-validation-toolkit``'s real Sharpe/PSR -- neither is
reimplemented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd
from factor_model.data.schemas import MarketObservation
from factor_model.regression import _period_realized_returns
from quant_toolkit.metrics import probabilistic_sharpe_ratio, sharpe_ratio

from ..data.schemas import OptimizationResult


@dataclass
class BacktestResult:
    gross_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    transaction_costs: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    net_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    cumulative_net_return: float = 0.0
    sharpe_ratio: float = float("nan")
    probabilistic_sharpe_ratio: float = float("nan")

    def to_dict(self) -> dict:
        return {
            "cumulative_net_return": self.cumulative_net_return,
            "sharpe_ratio": self.sharpe_ratio,
            "probabilistic_sharpe_ratio": self.probabilistic_sharpe_ratio,
            "n_periods": int(self.net_returns.notna().sum()),
            "net_returns": {str(k.date()): v for k, v in self.net_returns.items()},
        }


def run_backtest(results: Sequence[OptimizationResult], market_observations: Sequence[MarketObservation]) -> BacktestResult:
    """For each consecutive pair of real rebalance results (``results``
    sorted by ``as_of``), holds ``results[i]``'s target weights from
    ``results[i].as_of`` to ``results[i+1].as_of``, computes the real
    weighted realized return over that period (via factor-model-
    engine's own point-in-time ``_period_realized_returns``), and
    subtracts ``results[i].transaction_cost`` -- the real cost of
    ESTABLISHING those weights, paid at the start of the holding
    period they describe. A security with no real price at either end
    of the period is excluded from the weighted sum, consistent with
    ``_period_realized_returns``'s own point-in-time handling -- never
    silently treated as a 0% return.
    """
    sorted_results = sorted(results, key=lambda r: r.as_of)
    if len(sorted_results) < 2:
        raise ValueError("need at least 2 real rebalance results to measure a holding-period return")

    gross_by_period: dict[pd.Timestamp, float] = {}
    cost_by_period: dict[pd.Timestamp, float] = {}

    for start, end in zip(sorted_results[:-1], sorted_results[1:]):
        weights = start.target_weights
        security_ids = list(weights.keys())
        stock_returns = _period_realized_returns(market_observations, security_ids, start.as_of, end.as_of)
        gross = sum(weights[sid] * stock_returns[sid] for sid in security_ids if sid in stock_returns.index)
        gross_by_period[end.as_of] = gross
        cost_by_period[end.as_of] = start.transaction_cost

    gross_returns = pd.Series(gross_by_period).sort_index()
    transaction_costs = pd.Series(cost_by_period).sort_index()
    net_returns = gross_returns - transaction_costs
    cumulative_net_return = float((1.0 + net_returns).prod() - 1.0)

    return BacktestResult(
        gross_returns=gross_returns,
        transaction_costs=transaction_costs,
        net_returns=net_returns,
        cumulative_net_return=cumulative_net_return,
        sharpe_ratio=sharpe_ratio(net_returns),
        probabilistic_sharpe_ratio=probabilistic_sharpe_ratio(net_returns),
    )

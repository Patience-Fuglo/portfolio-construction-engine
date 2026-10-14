"""Stage 7 -- a real, deliberately simple transaction-cost model:
linear cost per unit of real turnover, plus an OPTIONAL market-impact
term scaling with trade size. This is a documented simplification, not
a full microstructure model -- no real bid-ask spread, no venue-
specific modeling, no participation-rate throttling. It exists to put
a genuine number on what every ``OptimizationResult.turnover`` already
tracks, and (optionally) to make the optimizer itself cost-aware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TransactionCostModel:
    # RESEARCH ASSUMPTION -- a simple, round-trip-ish linear cost per
    # unit of real turnover, in basis points
    linear_cost_bps: float = 10.0
    # RESEARCH ASSUMPTION -- additional cost proportional to
    # trade_size^1.5 (a common simple market-impact functional form);
    # 0.0 (the default) disables impact entirely -- a real, honest
    # default rather than an invented impact coefficient
    market_impact_coefficient: float = 0.0

    def __post_init__(self) -> None:
        if self.linear_cost_bps < 0:
            raise ValueError("linear_cost_bps cannot be negative")
        if self.market_impact_coefficient < 0:
            raise ValueError("market_impact_coefficient cannot be negative")

    def estimate_cost(self, target_weights: dict[str, float], current_weights: dict[str, float], security_ids: Sequence[str]) -> float:
        """Real estimated cost, as a fraction of portfolio value, of
        trading from ``current_weights`` to ``target_weights`` over
        ``security_ids``. A security absent from either dict is
        treated as a real 0.0 weight there (consistent with
        ``OptimizationInput.current_weight``'s own convention for a
        genuinely new or fully-exited position, not a missing
        observation).
        """
        total = 0.0
        for security_id in security_ids:
            trade_size = abs(target_weights.get(security_id, 0.0) - current_weights.get(security_id, 0.0))
            total += trade_size * (self.linear_cost_bps / 10_000.0)
            if self.market_impact_coefficient > 0:
                total += self.market_impact_coefficient * trade_size**1.5
        return total

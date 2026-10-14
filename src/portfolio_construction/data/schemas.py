"""Core data schemas for the portfolio construction engine.

Two structurally separate concepts, consistent with ``factor-model-
engine``'s own convention of never merging a point-in-time INPUT
snapshot with a derived OUTPUT:

1. ``OptimizationInput`` -- everything an optimizer needs, as of one
   real date: expected returns, a real stock-level covariance matrix,
   and the current (pre-trade) weights. A frozen, auditable snapshot.
2. ``OptimizationResult`` -- what an optimizer actually produced from
   that input: target weights, expected return/risk, turnover, and
   (critically) ``converged`` -- a real, independently-verified flag,
   never a blind pass-through of a solver's own ``success`` attribute.
   The risk-parity optimizer in this account's systematic-equity-
   research repo had a real bug once where a solver reported
   ``success: True`` while leaving weights at the unconverged starting
   guess; this schema's ``converged`` field exists specifically so that
   lesson can never silently regress here.

``PortfolioHolding`` (current/target weights) and ``FactorExposureRow``/
``FactorCovarianceResult``-shaped inputs are reused directly from
``factor_model`` rather than redefined -- see ``risk_model/`` for how
this repo builds a real stock-level covariance matrix from
factor-model-engine's own factor covariance and specific-risk outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class OptimizationInput:
    """A frozen, point-in-time snapshot of everything one optimization
    run needs. ``known_as_of`` is when this WHOLE input package (the
    alpha signal, the covariance estimate, and the current weights
    together) became usable -- a real, single vintage, since an
    optimizer run itself is one atomic real-world decision made at one
    real point in time, even though the underlying covariance may have
    been estimated over a long real historical window.
    """

    as_of: pd.Timestamp
    security_ids: tuple[str, ...]
    expected_returns: dict[str, float]
    covariance: pd.DataFrame
    current_weights: dict[str, float] = field(default_factory=dict)
    known_as_of: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        if self.known_as_of is not None and self.known_as_of < self.as_of:
            raise ValueError(f"known_as_of ({self.known_as_of}) cannot precede as_of ({self.as_of}) -- look-ahead")
        if set(self.expected_returns.keys()) != set(self.security_ids):
            raise ValueError("expected_returns must have exactly one real entry per security_id, no more, no fewer")
        if set(self.covariance.index) != set(self.security_ids) or set(self.covariance.columns) != set(self.security_ids):
            raise ValueError("covariance's real index/columns must exactly match security_ids")
        if self.covariance.shape[0] != self.covariance.shape[1]:
            raise ValueError("covariance must be a real square matrix")
        unknown_current_weights = set(self.current_weights.keys()) - set(self.security_ids)
        if unknown_current_weights:
            raise ValueError(f"current_weights references securities outside security_ids: {sorted(unknown_current_weights)}")

    def current_weight(self, security_id: str) -> float:
        """A security with no explicit current weight has a real,
        economically meaningful default of 0.0 -- a genuinely new
        position starting from nothing, not a missing observation
        being silently zero-filled."""
        return self.current_weights.get(security_id, 0.0)


@dataclass(frozen=True)
class OptimizationResult:
    """What an optimizer actually produced. ``converged`` is set by the
    optimizer's OWN post-solve verification (e.g. real constraint
    satisfaction, a real gradient/KKT check, or an explicit convergence
    tolerance check) -- never a blind pass-through of
    ``scipy.optimize``'s own ``success`` flag, which has been real,
    directly observed to be wrong before in this account's other work.
    """

    as_of: pd.Timestamp
    portfolio_id: str
    target_weights: dict[str, float]
    expected_return: float
    expected_variance: float
    turnover: float
    converged: bool
    binding_constraints: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "as_of": str(self.as_of.date()),
            "portfolio_id": self.portfolio_id,
            "target_weights": self.target_weights,
            "expected_return": self.expected_return,
            "expected_variance": self.expected_variance,
            "turnover": self.turnover,
            "converged": self.converged,
            "binding_constraints": list(self.binding_constraints),
            "warnings": list(self.warnings),
        }

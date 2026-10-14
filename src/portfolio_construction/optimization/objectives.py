"""Stage 4 -- real, numpy-vectorized objective functions the Stage 5
optimizer minimizes. Every function here operates on a plain weight
ARRAY (not a dict), in a caller-supplied ``security_ids`` order, since
``scipy.optimize`` needs array-in/scalar-out functions; ``weights_to_array``/
``array_to_weights`` are the one real place that order gets fixed.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def weights_to_array(weights: dict[str, float], security_ids: Sequence[str]) -> np.ndarray:
    return np.array([weights.get(sid, 0.0) for sid in security_ids], dtype=float)


def array_to_weights(array: np.ndarray, security_ids: Sequence[str]) -> dict[str, float]:
    return dict(zip(security_ids, array.tolist()))


def expected_portfolio_return(weights: np.ndarray, expected_returns: np.ndarray) -> float:
    return float(weights @ expected_returns)


def portfolio_variance(weights: np.ndarray, covariance: np.ndarray) -> float:
    return float(weights @ covariance @ weights)


def mean_variance_objective(weights: np.ndarray, expected_returns: np.ndarray, covariance: np.ndarray, risk_aversion: float) -> float:
    """To MINIMIZE: the negative of the real mean-variance utility
    ``expected_return - risk_aversion * variance``. Maximizing utility
    is equivalent to minimizing its negative -- ``scipy.optimize.minimize``
    always minimizes, so every objective in this module returns the
    quantity that function should actually drive down.
    """
    return -(expected_portfolio_return(weights, expected_returns) - risk_aversion * portfolio_variance(weights, covariance))


def min_variance_objective(weights: np.ndarray, covariance: np.ndarray) -> float:
    """The real minimum-variance objective -- mean-variance's special
    case as risk_aversion -> infinity, kept as its own clear function
    since a pure min-variance portfolio is a standard real mandate on
    its own, not just a limiting case."""
    return portfolio_variance(weights, covariance)


def risk_contributions(weights: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Each real asset's own contribution to total portfolio variance:
    ``w_i * (Sigma . w)_i``. Sums EXACTLY to the real total portfolio
    variance -- a genuine risk decomposition, not an approximation.
    """
    return weights * (covariance @ weights)


def risk_parity_objective(weights: np.ndarray, covariance: np.ndarray) -> float:
    """A RELATIVE risk-parity objective: ``sum((contribution_i / target
    - 1)^2)``, target = total_variance / n (equal risk contribution).

    Deliberately NOT an absolute squared difference
    (``sum((contribution_i - target)^2)``): that formulation's real
    scale is set by ``target`` itself, which can be small enough
    (given real covariance magnitudes) that a solver falsely reports
    convergence while leaving weights at the unconverged starting
    guess -- a real bug once caught in this account's
    systematic-equity-research risk-parity optimizer, verified by
    directly decomposing risk contributions at the reported
    "solution" and finding them up to 3x apart, not equal at all. The
    relative formulation here, plus ``is_risk_parity_converged``'s own
    independent post-solve check, is how that exact regression is
    prevented here.
    """
    total_variance = portfolio_variance(weights, covariance)
    if total_variance <= 0:
        return float("inf")
    n = len(weights)
    target = total_variance / n
    contributions = risk_contributions(weights, covariance)
    return float(np.sum((contributions / target - 1.0) ** 2))


def is_risk_parity_converged(weights: np.ndarray, covariance: np.ndarray, relative_tolerance: float = 0.05) -> bool:
    """The real, independent post-solve check: are every asset's ACTUAL
    risk contributions genuinely within ``relative_tolerance`` of equal
    -- never inferred from the solver's own ``success`` flag or from
    the objective value alone, both of which have been real, directly
    observed to be misleading before.
    """
    total_variance = portfolio_variance(weights, covariance)
    if total_variance <= 0:
        return False
    n = len(weights)
    target = total_variance / n
    contributions = risk_contributions(weights, covariance)
    relative_spread = float(np.max(np.abs(contributions / target - 1.0)))
    return relative_spread <= relative_tolerance

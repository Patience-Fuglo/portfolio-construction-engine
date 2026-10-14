"""Stage 3 -- real, composable portfolio constraints. Every constraint
implements the same ``check(weights, opt_input) -> list[ConstraintViolation]``
interface, so the optimizer (Stage 5) can run the exact same real
checks AFTER solving -- never trusting a solver's own ``success`` flag
alone, the lesson this whole engine's ``OptimizationResult.converged``
field exists to enforce. An empty list means genuinely satisfied
(within ``tolerance``), not merely "the solver didn't complain."
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..data.schemas import OptimizationInput

DEFAULT_TOLERANCE = 1e-8


@dataclass(frozen=True)
class ConstraintViolation:
    constraint_name: str
    detail: str
    magnitude: float  # strictly positive -- how far over the real limit


def _weight(weights: dict[str, float], security_id: str) -> float:
    return weights.get(security_id, 0.0)


@dataclass(frozen=True)
class MaxPositionWeightConstraint:
    """No single real security's weight may exceed ``max_weight`` in
    magnitude. ``long_only=True`` (the real default for most long-only
    mandates) additionally forbids any negative weight.
    """

    max_weight: float
    long_only: bool = True
    name: str = "max_position_weight"

    def check(self, weights: dict[str, float], opt_input: OptimizationInput, tolerance: float = DEFAULT_TOLERANCE) -> list[ConstraintViolation]:
        violations = []
        for security_id in opt_input.security_ids:
            w = _weight(weights, security_id)
            if self.long_only and w < -tolerance:
                violations.append(ConstraintViolation(self.name, f"{security_id}: weight {w:.6f} < 0 (long-only)", -w))
                continue
            if abs(w) > self.max_weight + tolerance:
                violations.append(
                    ConstraintViolation(self.name, f"{security_id}: |weight| {abs(w):.6f} exceeds {self.max_weight}", abs(w) - self.max_weight)
                )
        return violations


@dataclass(frozen=True)
class SectorExposureConstraint:
    """No real sector's NET weight (sum of its constituents' weights)
    may exceed ``max_abs_sector_weight`` in magnitude.
    """

    sector_by_security: dict[str, str]
    max_abs_sector_weight: float
    name: str = "sector_exposure"

    def check(self, weights: dict[str, float], opt_input: OptimizationInput, tolerance: float = DEFAULT_TOLERANCE) -> list[ConstraintViolation]:
        sector_totals: dict[str, float] = {}
        for security_id in opt_input.security_ids:
            sector = self.sector_by_security.get(security_id, "unknown")
            sector_totals[sector] = sector_totals.get(sector, 0.0) + _weight(weights, security_id)
        violations = []
        for sector, total in sorted(sector_totals.items()):
            if abs(total) > self.max_abs_sector_weight + tolerance:
                violations.append(
                    ConstraintViolation(self.name, f"{sector}: net weight {total:.6f} exceeds {self.max_abs_sector_weight}", abs(total) - self.max_abs_sector_weight)
                )
        return violations


@dataclass(frozen=True)
class FactorExposureConstraint:
    """No real factor's NET portfolio exposure (``w . exposure_matrix[:,
    factor]``) may exceed ``max_abs_exposure[factor]`` in magnitude.
    ``exposure_matrix`` is the real stock x factor matrix from
    ``risk_model.build_exposure_matrix`` -- never re-derived here.
    """

    exposure_matrix: pd.DataFrame  # security_id index, factor_name columns
    max_abs_exposure: dict[str, float]
    name: str = "factor_exposure"

    def check(self, weights: dict[str, float], opt_input: OptimizationInput, tolerance: float = DEFAULT_TOLERANCE) -> list[ConstraintViolation]:
        violations = []
        for factor_name, limit in sorted(self.max_abs_exposure.items()):
            if factor_name not in self.exposure_matrix.columns:
                continue
            column = self.exposure_matrix[factor_name]
            net_exposure = sum(_weight(weights, sid) * column.get(sid, 0.0) for sid in opt_input.security_ids)
            if abs(net_exposure) > limit + tolerance:
                violations.append(
                    ConstraintViolation(self.name, f"{factor_name}: net exposure {net_exposure:.6f} exceeds {limit}", abs(net_exposure) - limit)
                )
        return violations


@dataclass(frozen=True)
class TurnoverConstraint:
    """Total one-way turnover (``sum(|target - current|)``) may not
    exceed ``max_turnover`` -- a real, direct trading-cost/capacity limit.
    """

    max_turnover: float
    name: str = "turnover"

    def check(self, weights: dict[str, float], opt_input: OptimizationInput, tolerance: float = DEFAULT_TOLERANCE) -> list[ConstraintViolation]:
        turnover = sum(abs(_weight(weights, sid) - opt_input.current_weight(sid)) for sid in opt_input.security_ids)
        if turnover > self.max_turnover + tolerance:
            return [ConstraintViolation(self.name, f"total turnover {turnover:.6f} exceeds {self.max_turnover}", turnover - self.max_turnover)]
        return []


@dataclass(frozen=True)
class GrossLeverageConstraint:
    """Total gross exposure (``sum(|w_i|)``) may not exceed
    ``max_gross_leverage`` (1.0 for a fully-invested long-only book with
    no leverage; > 1.0 for a real levered or long/short book).
    """

    max_gross_leverage: float
    name: str = "gross_leverage"

    def check(self, weights: dict[str, float], opt_input: OptimizationInput, tolerance: float = DEFAULT_TOLERANCE) -> list[ConstraintViolation]:
        gross = sum(abs(_weight(weights, sid)) for sid in opt_input.security_ids)
        if gross > self.max_gross_leverage + tolerance:
            return [ConstraintViolation(self.name, f"gross leverage {gross:.6f} exceeds {self.max_gross_leverage}", gross - self.max_gross_leverage)]
        return []


def check_all(constraints, weights: dict[str, float], opt_input: OptimizationInput, tolerance: float = DEFAULT_TOLERANCE) -> list[ConstraintViolation]:
    """Run every real constraint's own ``check`` and flatten the result
    -- the one function the optimizer's post-solve verification (Stage
    5) actually calls, so there is exactly one real place that decides
    "is this candidate portfolio genuinely feasible."
    """
    violations: list[ConstraintViolation] = []
    for constraint in constraints:
        violations.extend(constraint.check(weights, opt_input, tolerance))
    return violations

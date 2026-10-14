import pandas as pd
import pytest

from portfolio_construction.constraints import (
    FactorExposureConstraint,
    GrossLeverageConstraint,
    MaxPositionWeightConstraint,
    SectorExposureConstraint,
    TurnoverConstraint,
    check_all,
)
from portfolio_construction.data import OptimizationInput


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _opt_input(security_ids, current_weights=None) -> OptimizationInput:
    import numpy as np

    n = len(security_ids)
    covariance = pd.DataFrame(np.eye(n) * 0.04, index=security_ids, columns=security_ids)
    return OptimizationInput(
        as_of=_ts("2020-06-30"), security_ids=tuple(security_ids),
        expected_returns={sid: 0.01 for sid in security_ids}, covariance=covariance,
        current_weights=current_weights or {},
    )


def test_max_position_weight_flags_an_oversized_real_position():
    opt_input = _opt_input(["A", "B"])
    constraint = MaxPositionWeightConstraint(max_weight=0.3)
    violations = constraint.check({"A": 0.5, "B": 0.5}, opt_input)
    assert len(violations) == 2
    assert violations[0].magnitude == pytest.approx(0.2)


def test_max_position_weight_long_only_flags_a_short_position():
    opt_input = _opt_input(["A"])
    constraint = MaxPositionWeightConstraint(max_weight=1.0, long_only=True)
    violations = constraint.check({"A": -0.1}, opt_input)
    assert len(violations) == 1
    assert violations[0].magnitude == pytest.approx(0.1)


def test_max_position_weight_allows_short_positions_when_not_long_only():
    opt_input = _opt_input(["A"])
    constraint = MaxPositionWeightConstraint(max_weight=0.5, long_only=False)
    assert constraint.check({"A": -0.3}, opt_input) == []


def test_sector_exposure_flags_a_real_sector_concentration():
    opt_input = _opt_input(["A", "B", "C"])
    constraint = SectorExposureConstraint(sector_by_security={"A": "tech", "B": "tech", "C": "energy"}, max_abs_sector_weight=0.4)
    violations = constraint.check({"A": 0.3, "B": 0.3, "C": 0.1}, opt_input)
    assert len(violations) == 1
    assert "tech" in violations[0].detail
    assert violations[0].magnitude == pytest.approx(0.2)


def test_factor_exposure_flags_a_real_excess_factor_tilt():
    opt_input = _opt_input(["A", "B"])
    exposure_matrix = pd.DataFrame({"size": [2.0, -1.0]}, index=["A", "B"])
    constraint = FactorExposureConstraint(exposure_matrix=exposure_matrix, max_abs_exposure={"size": 0.5})
    violations = constraint.check({"A": 0.5, "B": 0.5}, opt_input)
    # net size exposure = 0.5*2.0 + 0.5*(-1.0) = 0.5 -- right at the limit, not over
    assert violations == []
    violations_over = constraint.check({"A": 0.8, "B": 0.2}, opt_input)
    # net = 0.8*2.0 + 0.2*(-1.0) = 1.4
    assert len(violations_over) == 1
    assert violations_over[0].magnitude == pytest.approx(0.9)


def test_turnover_constraint_flags_real_excess_trading():
    opt_input = _opt_input(["A", "B"], current_weights={"A": 0.5, "B": 0.5})
    constraint = TurnoverConstraint(max_turnover=0.2)
    violations = constraint.check({"A": 0.9, "B": 0.1}, opt_input)
    # turnover = |0.9-0.5| + |0.1-0.5| = 0.8
    assert len(violations) == 1
    assert violations[0].magnitude == pytest.approx(0.6)


def test_gross_leverage_flags_a_real_over_levered_book():
    opt_input = _opt_input(["A", "B"])
    constraint = GrossLeverageConstraint(max_gross_leverage=1.0)
    violations = constraint.check({"A": 0.8, "B": -0.8}, opt_input)
    assert len(violations) == 1
    assert violations[0].magnitude == pytest.approx(0.6)


def test_check_all_aggregates_violations_from_every_real_constraint():
    opt_input = _opt_input(["A", "B"])
    constraints = [MaxPositionWeightConstraint(max_weight=0.3), GrossLeverageConstraint(max_gross_leverage=0.5)]
    violations = check_all(constraints, {"A": 0.6, "B": 0.4}, opt_input)
    assert len(violations) >= 2
    assert {v.constraint_name for v in violations} == {"max_position_weight", "gross_leverage"}


def test_check_all_returns_empty_for_a_genuinely_feasible_portfolio():
    opt_input = _opt_input(["A", "B"])
    constraints = [MaxPositionWeightConstraint(max_weight=0.6), GrossLeverageConstraint(max_gross_leverage=1.0)]
    assert check_all(constraints, {"A": 0.5, "B": 0.5}, opt_input) == []

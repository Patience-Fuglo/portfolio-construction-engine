import pandas as pd
import pytest

from portfolio_construction.data import OptimizationInput, OptimizationResult


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _covariance(security_ids) -> pd.DataFrame:
    n = len(security_ids)
    import numpy as np

    values = np.eye(n) * 0.04
    return pd.DataFrame(values, index=security_ids, columns=security_ids)


def test_optimization_input_accepts_a_consistent_real_universe():
    sids = ("A", "B", "C")
    opt_input = OptimizationInput(
        as_of=_ts("2020-06-30"), security_ids=sids,
        expected_returns={"A": 0.01, "B": 0.02, "C": -0.01},
        covariance=_covariance(sids), known_as_of=_ts("2020-06-30"),
    )
    assert opt_input.current_weight("A") == 0.0  # real default: a new position, not missing data


def test_optimization_input_rejects_known_as_of_before_as_of():
    sids = ("A",)
    with pytest.raises(ValueError):
        OptimizationInput(
            as_of=_ts("2020-06-30"), security_ids=sids, expected_returns={"A": 0.01},
            covariance=_covariance(sids), known_as_of=_ts("2020-06-29"),
        )


def test_optimization_input_allows_known_as_of_to_be_unset():
    sids = ("A",)
    opt_input = OptimizationInput(
        as_of=_ts("2020-06-30"), security_ids=sids, expected_returns={"A": 0.01}, covariance=_covariance(sids),
    )
    assert opt_input.known_as_of is None


def test_optimization_input_rejects_expected_returns_missing_a_security():
    sids = ("A", "B")
    with pytest.raises(ValueError):
        OptimizationInput(
            as_of=_ts("2020-06-30"), security_ids=sids, expected_returns={"A": 0.01},
            covariance=_covariance(sids),
        )


def test_optimization_input_rejects_expected_returns_with_an_extra_security():
    sids = ("A",)
    with pytest.raises(ValueError):
        OptimizationInput(
            as_of=_ts("2020-06-30"), security_ids=sids, expected_returns={"A": 0.01, "B": 0.02},
            covariance=_covariance(sids),
        )


def test_optimization_input_rejects_covariance_with_mismatched_universe():
    sids = ("A", "B")
    with pytest.raises(ValueError):
        OptimizationInput(
            as_of=_ts("2020-06-30"), security_ids=sids, expected_returns={"A": 0.01, "B": 0.02},
            covariance=_covariance(("A", "C")),
        )


def test_optimization_input_rejects_current_weights_outside_the_universe():
    sids = ("A",)
    with pytest.raises(ValueError):
        OptimizationInput(
            as_of=_ts("2020-06-30"), security_ids=sids, expected_returns={"A": 0.01},
            covariance=_covariance(sids), current_weights={"Z": 0.5},
        )


def test_optimization_result_to_dict_round_trips_the_real_fields():
    result = OptimizationResult(
        as_of=_ts("2020-06-30"), portfolio_id="P", target_weights={"A": 0.5, "B": 0.5},
        expected_return=0.01, expected_variance=0.02, turnover=0.3, converged=True,
        binding_constraints=("max_position_weight",), warnings=(),
    )
    payload = result.to_dict()
    assert payload["converged"] is True
    assert payload["binding_constraints"] == ["max_position_weight"]
    assert payload["target_weights"] == {"A": 0.5, "B": 0.5}

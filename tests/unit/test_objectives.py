import numpy as np
import pytest

from portfolio_construction.optimization import (
    array_to_weights,
    is_risk_parity_converged,
    mean_variance_objective,
    min_variance_objective,
    portfolio_variance,
    risk_contributions,
    risk_parity_objective,
    weights_to_array,
)


def test_weights_to_array_and_back_round_trip_in_order():
    weights = {"B": 0.3, "A": 0.7}
    array = weights_to_array(weights, ["A", "B"])
    np.testing.assert_allclose(array, [0.7, 0.3])
    assert array_to_weights(array, ["A", "B"]) == {"A": 0.7, "B": 0.3}


def test_weights_to_array_defaults_a_missing_security_to_zero():
    array = weights_to_array({"A": 0.5}, ["A", "B"])
    np.testing.assert_allclose(array, [0.5, 0.0])


def test_portfolio_variance_matches_the_real_quadratic_form():
    w = np.array([0.6, 0.4])
    cov = np.array([[0.04, 0.01], [0.01, 0.02]])
    assert portfolio_variance(w, cov) == pytest.approx(float(w @ cov @ w))


def test_mean_variance_objective_rewards_higher_return_lower_risk():
    cov = np.eye(2) * 0.02
    returns = np.array([0.1, 0.1])
    w_balanced = np.array([0.5, 0.5])
    w_concentrated = np.array([1.0, 0.0])
    # same expected return either way (symmetric returns), but
    # concentrated has HIGHER variance -- the minimized objective
    # (negative utility) should be larger (worse) for concentrated
    obj_balanced = mean_variance_objective(w_balanced, returns, cov, risk_aversion=5.0)
    obj_concentrated = mean_variance_objective(w_concentrated, returns, cov, risk_aversion=5.0)
    assert obj_concentrated > obj_balanced


def test_min_variance_objective_is_just_the_real_variance():
    w = np.array([0.5, 0.5])
    cov = np.array([[0.04, 0.0], [0.0, 0.01]])
    assert min_variance_objective(w, cov) == pytest.approx(portfolio_variance(w, cov))


def test_risk_contributions_sum_exactly_to_total_variance():
    w = np.array([0.3, 0.7])
    cov = np.array([[0.05, 0.02], [0.02, 0.03]])
    contributions = risk_contributions(w, cov)
    assert float(contributions.sum()) == pytest.approx(portfolio_variance(w, cov))


def test_risk_parity_objective_is_zero_for_a_genuinely_equal_risk_portfolio():
    # a diagonal covariance with equal variances -- equal weights give
    # exactly equal risk contributions
    cov = np.eye(3) * 0.02
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    assert risk_parity_objective(w, cov) == pytest.approx(0.0, abs=1e-10)


def test_risk_parity_objective_is_positive_for_an_unequal_risk_portfolio():
    cov = np.diag([0.01, 0.09])  # asset 1 three times the variance of asset 0
    w = np.array([0.5, 0.5])  # equal WEIGHT, but very unequal RISK
    assert risk_parity_objective(w, cov) > 0.1


def test_is_risk_parity_converged_catches_the_real_historical_bug():
    # the exact real bug once caught in systematic-equity-research: a
    # solver reporting "success" while weights sit at equal-weight on a
    # covariance where that is NOT the real risk-parity solution
    cov = np.diag([0.01, 0.01, 0.09])  # asset 2 nine times the variance
    equal_weight = np.array([1 / 3, 1 / 3, 1 / 3])
    assert not is_risk_parity_converged(equal_weight, cov, relative_tolerance=0.05)


def test_is_risk_parity_converged_accepts_the_real_analytic_solution():
    # for a diagonal covariance, the real risk-parity solution is
    # w_i proportional to 1/sqrt(variance_i)
    variances = np.array([0.01, 0.04, 0.09])
    cov = np.diag(variances)
    raw = 1.0 / np.sqrt(variances)
    w = raw / raw.sum()
    assert is_risk_parity_converged(w, cov, relative_tolerance=1e-6)

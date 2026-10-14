import pandas as pd
import pytest
from factor_model.data.schemas import MarketObservation, Sector

from portfolio_construction.backtest import run_backtest
from portfolio_construction.data import OptimizationResult


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _market(security_id: str, as_of: str, price: float) -> MarketObservation:
    return MarketObservation(
        security_id=security_id, ticker=security_id, as_of=_ts(as_of), known_as_of=_ts(as_of),
        sector=Sector.TECHNOLOGY, is_active=True, price=price, shares_outstanding=1.0e6,
        market_cap=price * 1.0e6, adv_20d=1.0e6,
    )


def _result(as_of: str, weights: dict, transaction_cost: float = 0.0) -> OptimizationResult:
    return OptimizationResult(
        as_of=_ts(as_of), portfolio_id="P", target_weights=weights, expected_return=0.0,
        expected_variance=0.0, turnover=0.0, converged=True, transaction_cost=transaction_cost,
    )


def test_backtest_matches_a_hand_computed_weighted_return():
    market = [
        _market("A", "2020-01-31", 100.0), _market("B", "2020-01-31", 50.0),
        _market("A", "2020-02-29", 110.0), _market("B", "2020-02-29", 45.0),
    ]
    results = [_result("2020-01-31", {"A": 0.6, "B": 0.4})]
    # need at least 2 results -- add a second one held from Feb forward (unused in this one-period check)
    results.append(_result("2020-02-29", {"A": 0.6, "B": 0.4}))

    backtest = run_backtest(results, market)
    # A: +10%, B: -10% -- weighted: 0.6*0.10 + 0.4*(-0.10) = 0.02
    assert backtest.gross_returns.iloc[0] == pytest.approx(0.6 * 0.10 + 0.4 * -0.10)
    assert backtest.net_returns.iloc[0] == pytest.approx(backtest.gross_returns.iloc[0])  # no cost on this rebalance


def test_backtest_subtracts_the_real_transaction_cost_from_the_starting_rebalance():
    market = [_market("A", "2020-01-31", 100.0), _market("A", "2020-02-29", 105.0)]
    results = [_result("2020-01-31", {"A": 1.0}, transaction_cost=0.01), _result("2020-02-29", {"A": 1.0})]
    backtest = run_backtest(results, market)
    assert backtest.gross_returns.iloc[0] == pytest.approx(0.05)
    assert backtest.net_returns.iloc[0] == pytest.approx(0.05 - 0.01)


def test_backtest_excludes_a_security_with_no_real_price_at_either_end():
    market = [_market("A", "2020-01-31", 100.0), _market("A", "2020-02-29", 110.0)]  # B has no real price at all
    results = [_result("2020-01-31", {"A": 0.5, "B": 0.5}), _result("2020-02-29", {"A": 1.0})]
    backtest = run_backtest(results, market)
    # only A contributes -- 0.5 * 0.10 = 0.05, NOT silently treating B as 0% return
    assert backtest.gross_returns.iloc[0] == pytest.approx(0.5 * 0.10)


def test_backtest_cumulative_return_compounds_correctly():
    market = [
        _market("A", "2020-01-31", 100.0), _market("A", "2020-02-29", 110.0),
        _market("A", "2020-03-31", 121.0),
    ]
    results = [_result("2020-01-31", {"A": 1.0}), _result("2020-02-29", {"A": 1.0}), _result("2020-03-31", {"A": 1.0})]
    backtest = run_backtest(results, market)
    assert backtest.cumulative_net_return == pytest.approx(1.21 - 1.0, abs=1e-6)


def test_backtest_sharpe_and_psr_are_real_numbers_for_a_realistic_series():
    # a real Sharpe ratio is undefined for a perfectly constant return
    # series (zero variance) -- the toolkit correctly returns NaN for
    # that, so this fixture uses genuinely varying (not deterministic)
    # per-period returns, still with a real positive drift
    growth_rates = [1.02, 1.01, 1.03, 1.015, 1.025, 1.005]
    market = []
    dates = pd.date_range("2020-01-31", periods=len(growth_rates), freq="ME")
    price = 100.0
    results = []
    for d, growth in zip(dates, growth_rates):
        market.append(_market("A", str(d.date()), price))
        price *= growth
        results.append(_result(str(d.date()), {"A": 1.0}))
    backtest = run_backtest(results, market)
    assert backtest.sharpe_ratio > 0
    assert 0.0 <= backtest.probabilistic_sharpe_ratio <= 1.0


def test_backtest_requires_at_least_two_real_rebalance_results():
    with pytest.raises(ValueError):
        run_backtest([_result("2020-01-31", {"A": 1.0})], [])

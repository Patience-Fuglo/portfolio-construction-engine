# Build status

Tracks stage-by-stage completion, mirroring `factor-model-engine`'s own
`docs/STATUS.md` convention, so work is resumable across sessions.

## Stage 1 -- Foundation: COMPLETE

`OptimizationInput`/`OptimizationResult` schemas (`data/schemas.py`),
`EngineConfig`/`load_config` (`config.py`), dependency wiring to
`factor-model-engine` (pinned git dependency, confirmed resolving
cleanly in a fresh venv) and `alpha-validation-toolkit`. 8 tests
passing. Pushed.

## Stage 2 -- Risk Model: COMPLETE

Added `risk_model/stock_covariance.py`: `build_exposure_matrix`
(pivots factor-model-engine's stock-level `FactorExposureRow`s into
the real `B` matrix) and `build_stock_covariance_matrix` (the real
Barra-style decomposition `Cov = B·Σ_f·Bᵗ + diag(specific_variance)`).
Two explicit, warned (never silent) assumptions where a real input is
missing: a missing exposure cell contributes 0 (neutral, since
standardized exposures are zero-mean); a missing specific-variance
estimate is floored at `RiskModelConfig.min_specific_variance`, never
fabricated as exactly 0 idiosyncratic risk.

5 new unit tests (including a real symmetric/positive-semidefinite
check) plus a genuine cross-repo integration test that builds a stock
covariance matrix directly from factor-model-engine's own real
pipeline output (universe, exposures, factor returns, factor
covariance, specific risk) on its synthetic dataset -- not a hand-built
stand-in for the dependency. 14 tests passing (was 8). Pushed.

## Stage 3 -- Constraints: COMPLETE

Added `constraints/constraints.py`: five real, composable constraint
types -- `MaxPositionWeightConstraint`, `SectorExposureConstraint`,
`FactorExposureConstraint` (using `risk_model.build_exposure_matrix`'s
own output, never re-derived), `TurnoverConstraint`, and
`GrossLeverageConstraint` -- each implementing the same
`check(weights, opt_input) -> list[ConstraintViolation]` interface,
plus `check_all` to run every constraint and flatten the result. An
empty list means genuinely satisfied within tolerance, not merely "the
solver didn't complain" -- this is the exact function Stage 5's
optimizer will call for real post-solve verification, never trusting
a solver's own `success` flag alone.

9 new unit tests covering every constraint type's real violation
magnitude and the aggregate `check_all` path. 23 tests passing (was
14). Pushed.

## Stage 4 -- Objectives: COMPLETE

Added `optimization/objectives.py`: numpy-vectorized `mean_variance_objective`,
`min_variance_objective`, `risk_parity_objective` (and the shared
`portfolio_variance`/`risk_contributions`/`weights_to_array`/
`array_to_weights` helpers Stage 5's optimizer will reuse).

`risk_parity_objective` directly ports the real fix from this
account's `systematic-equity-research` risk-parity optimizer bug: a
RELATIVE objective (`sum((contribution/target - 1)^2)`), never an
absolute squared difference, whose real scale could let a solver
falsely report convergence at the unconverged starting guess.
`is_risk_parity_converged` is the independent, real post-solve
verification that bug fix also required -- checking ACTUAL risk
contributions directly, never inferred from a solver's own `success`
flag or the objective value alone.

10 new unit tests, including one that directly reconstructs the exact
historical bug scenario (equal weights on an unequal-variance diagonal
covariance correctly fails convergence) and one that accepts the real
analytic risk-parity solution for a diagonal covariance. 33 tests
passing (was 23). Pushed.

## Stage 5 -- Optimizer: COMPLETE

Added `optimization/optimizer.py` (`optimize_portfolio`): `scipy.optimize.
minimize` (SLSQP) wrapped around the three Stage 4 objectives, with
real per-type scipy constraint translation (position limits become
bounds; sector/factor/turnover/leverage become inequality constraints;
a fully-invested or dollar-neutral equality constraint via
`OptimizerConfig.target_net_exposure`).

**The central discipline, enforced directly, not just documented:**
after the solver returns, `OptimizationResult.converged` is set from
an INDEPENDENT real check -- `constraints.check_all` against the
actual returned weights, plus `is_risk_parity_converged` for the
risk-parity objective -- never from `scipy`'s own `success` flag. Both
real disagreement directions are explicitly warned about: a scipy
"success" that the independent check finds infeasible (the exact
historical failure mode), and a scipy "failure" that the independent
check finds genuinely feasible anyway.

6 new unit tests, including one that reconstructs a real infeasible
constraint combination (`max_position_weight=0.3` over 2 long-only
securities makes the fully-invested equality mathematically
impossible) and confirms the optimizer correctly reports non-convergence
rather than returning an infeasible "solution" silently. Plus a
cross-repo integration test: min-variance optimization against a
stock covariance matrix built directly from factor-model-engine's own
real pipeline output. 40 tests passing (was 33).

One documented, honest simplification: `binding_constraints` reports
VIOLATED constraint names when not converged (for triage), not
genuinely active-but-feasible ones -- a real margin-based "is this
constraint exactly at its limit" check is left for a later refinement,
not claimed now.

## Stage 6 -- Rebalancing: COMPLETE

Added `rebalancing/rebalancer.py` (`run_rebalancing_schedule`): at
each real rebalance date, re-estimates the whole risk model (factor
exposures, factor covariance, specific risk) using only
`lookback_periods` of REAL trailing history ending at that date --
never a later one -- rebuilds the stock covariance matrix, calls the
real optimizer, and carries the result's target weights forward as
the NEXT rebalance's real pre-trade weights, so turnover constraints
see genuine trading continuity rather than a fresh start every time.

Alpha (expected returns) is deliberately an explicit per-date input
(`ExpectedReturnsSource`), not generated here -- signal research is a
different repo's real job; this engine takes it as given.

2 new integration tests on the real synthetic dataset: a real
quarterly schedule (confirming the first rebalance's turnover equals a
full build-from-nothing, and every rebalance converges) and a real
insufficient-history case (warns per-date rather than crashing). One
real test-fixture mistake caught and fixed along the way (a 5-period
synthetic config using the default `restatement_period_index=23`,
out of bounds for that short a history -- not a bug in either engine,
just a misconfigured fixture). 42 tests passing (was 40). Pushed.

## Stage 7 -- Transaction Costs: COMPLETE

Added `backtest/transaction_costs.py` (`TransactionCostModel`): a real,
deliberately simple linear-cost-per-unit-turnover model, plus an
optional market-impact term (`coefficient * trade_size^1.5`, 0 by
default -- a real, honest default rather than an invented impact
coefficient). Documented explicitly as a simplification: no real
bid-ask spread, no venue modeling, no participation-rate throttling.

Wired additively into Stage 5's `optimize_portfolio` via a new,
backward-compatible `cost_model` parameter (default `None`, matching
every existing call site's prior behavior exactly): when supplied with
`objective="mean_variance"`, the real estimated cost is SUBTRACTED
from the optimizer's own utility, so the optimizer genuinely trades
off expected alpha against the real cost of getting there, not just
applied after the fact. `OptimizationResult` gains a `transaction_cost`
field (defaulted to 0.0, backward compatible).

7 new unit tests, including one that directly demonstrates the real
behavioral effect: with a real alpha edge favoring a trade, a
high-cost model produces genuinely LESS turnover than no cost model at
all -- confirming the optimizer is actually using the cost signal, not
just reporting a number after the fact. 49 tests passing (was 42).
Pushed.

## Stage 8 -- Backtest: COMPLETE

Added `backtest/backtest.py` (`run_backtest`): forward-applies a real
rebalancing schedule's target weights, holding each one from its own
`as_of` to the next, computing the real weighted realized return per
period via factor-model-engine's own point-in-time
`_period_realized_returns` (never re-derived), net of the real
`transaction_cost` already reported on the rebalance that established
those weights. Reuses `alpha-validation-toolkit`'s real
`sharpe_ratio`/`probabilistic_sharpe_ratio` for evaluation -- neither
reimplemented.

`rebalancing/rebalancer.py` extended (additively, backward compatible)
with a `cost_model` parameter, threaded straight through to
`optimize_portfolio` at every rebalance date, so a real cost model
applies consistently across a whole schedule.

6 new unit tests (including a security with no real price at either
end of a period correctly excluded, never treated as a 0% return) and
2 integration tests on the real synthetic dataset -- including a
direct, real demonstration that a costed rebalancing schedule's
cumulative net return comes out genuinely lower than the same schedule
run with no cost model at all. 57 tests passing (was 49). Pushed.

## Stage 9 -- Reporting: COMPLETE

Added `reporting/report.py` (`report_to_dict`/`render_markdown_report`):
both derived from the SAME real `RebalancingSchedule` (Stage 6) and
`BacktestResult` (Stage 8) -- no new orchestration layer needed, since
`rebalancing/rebalancer.py` already fulfills that role (unlike
factor-model-engine, which needed a dedicated `pipeline/` module).
Mirrors factor-model-engine's own Stage 9 reporting convention.

4 new integration tests on the real synthetic dataset: JSON
serializability, reconciliation against the real underlying schedule/
backtest objects, every real section present in the rendered
Markdown, and deterministic rendering across two calls. 61 tests
passing (was 57). Pushed.

## Next up: Stage 10 -- Hardening

The comprehensive point-in-time check and full CLI-chain test,
mirroring `factor-model-engine`'s own Stage 10.

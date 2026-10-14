# Data dictionary

Every schema field this engine actually defines, plus the real formula
each build stage computes -- matching `factor-model-engine`'s own
`docs/data_dictionary.md` convention. Fields reused directly from
`factor-model-engine` (e.g. `FactorExposureRow`) are not re-documented
here; see that repo's own dictionary.

## `OptimizationInput` (`data/schemas.py`)

One real optimization request, fully self-contained.

- `as_of` -- the date this optimization is run for; every other PIT
  rule in this engine is relative to this date.
- `security_ids` -- the real tradable universe for this request, as an
  ordered tuple (array ordering is derived from this, everywhere).
- `expected_returns` -- `dict[security_id, float]`; `__post_init__`
  requires its keys to match `security_ids` exactly (no silent subset,
  no silent extra).
- `covariance` -- a `pandas.DataFrame` indexed and columned by
  `security_ids`, exactly, in that order; `__post_init__` requires it
  square and index/columns matching.
- `current_weights` -- `dict[security_id, float]`; `__post_init__`
  requires its keys to be a subset of `security_ids` (a security can be
  silently absent -- see `current_weight()` below -- but never present
  under an unknown id).
- `known_as_of` -- when this specific optimization input (its expected
  returns, covariance, and current weights) became knowable;
  `__post_init__` requires `known_as_of >= as_of`, the same real
  look-ahead guard every other engine in this portfolio enforces.

**`current_weight(security_id)`** -- returns `current_weights.get(security_id,
0.0)`. A real semantic choice, not a convenience default: a security
absent from `current_weights` is a genuinely new (zero) position, not a
missing observation -- this is the one place in the schema where
"absent" means a real zero, not "unknown."

## `OptimizationResult` (`data/schemas.py`)

One real optimization outcome.

- `as_of`, `portfolio_id` -- identifies which request and which real
  book this result belongs to.
- `target_weights` -- `dict[security_id, float]`, the real solved
  weights.
- `expected_return`, `expected_variance` -- `w @ mu`, `w @ Cov @ w` at
  the real solved `w`.
- `turnover` -- `sum(abs(target - current))` across all `security_ids`.
- `converged` -- **never** `scipy`'s own `success` flag; see the
  Stage 5 formula section below for exactly how this is independently
  derived.
- `binding_constraints` -- when `converged` is `False`, the names of
  the constraints that were actually **violated** (for triage), not
  genuinely "active but still feasible" ones -- this v1 does not yet
  distinguish the two; see `optimizer.py`'s own docstring.
- `warnings` -- every real diagnostic message produced during the
  solve (constraint violations, net-exposure mismatch, risk-parity
  non-convergence, or a disagreement between `scipy`'s own flag and
  the independent check).
- `transaction_cost` -- defaults to `0.0`; only populated when a real
  `TransactionCostModel` (Stage 7) is supplied to `optimize_portfolio`.

**`to_dict()`** -- a flat real serialization of all the above fields,
for reporting.

## Stage 2 -- stock covariance (`risk_model/stock_covariance.py`)

`build_exposure_matrix(stock_exposure_rows)` pivots `factor-model-engine`'s
own real `FactorExposureRow` sequence (filtered to `subject_type ==
STOCK`) into a security x factor matrix.

`build_stock_covariance_matrix(exposure_matrix, factor_covariance,
specific_variance, risk_cfg)` computes the real formula

```
Cov = B · Sigma_f · B^T + diag(specific_variance)
```

where `B` is the exposure matrix, `Sigma_f` is `factor-model-engine`'s
own real factor covariance, and `specific_variance` is its own real
per-security idiosyncratic variance. Two real, documented missing-data
behaviors (both warned, never silent):

- a security/factor pair absent from `exposure_matrix` -- treated as
  `0.0` exposure (factor-neutral on that factor, not an error).
- a security absent from `specific_variance` -- floored at
  `risk_cfg.min_specific_variance` (see
  [`research_assumptions.md`](research_assumptions.md)), never
  defaulted to exactly `0.0`.

## Stage 3 -- constraints (`constraints/constraints.py`)

`ConstraintViolation(constraint_name, detail, magnitude)` -- the real,
uniform violation record every constraint's own `check()` returns a
list of. `DEFAULT_TOLERANCE = 1e-8`.

Five real constraint types, each with its own `check(weights, opt_input,
tolerance) -> list[ConstraintViolation]`:

- **`MaxPositionWeightConstraint`**(`max_weight`, `long_only`) -- every
  individual `|weight| <= max_weight` (or `weight <= max_weight` and
  `weight >= 0` when `long_only`).
- **`SectorExposureConstraint`**(`sector_by_security`, `max_abs_sector_weight`)
  -- `|sum(weights in sector)| <= max_abs_sector_weight`, per real
  sector.
- **`FactorExposureConstraint`**(`exposure_matrix`, `max_abs_exposure`)
  -- `|sum(weight_i * exposure_i,factor)| <= max_abs_exposure[factor]`,
  per real factor.
- **`TurnoverConstraint`**(`max_turnover`) -- `sum(|weight -
  current_weight|) <= max_turnover`.
- **`GrossLeverageConstraint`**(`max_gross_leverage`) --
  `sum(|weight|) <= max_gross_leverage`.

`check_all(constraints, weights, opt_input, tolerance)` -- real
aggregator, concatenates every constraint's own violations.

## Stage 4 -- objectives (`optimization/objectives.py`)

- `expected_portfolio_return(w, mu) = w @ mu`.
- `portfolio_variance(w, Cov) = w @ Cov @ w`.
- `mean_variance_objective(w, mu, Cov, risk_aversion)` -- the real
  negated utility `-(w@mu - risk_aversion * w@Cov@w)`, for `scipy`'s
  own `minimize` (which only minimizes).
- `min_variance_objective(w, Cov) = w @ Cov @ w` (no return term).
- `risk_contributions(w, Cov)` -- real per-security `w_i * (Cov @ w)_i`,
  which sums exactly to `w @ Cov @ w` (verified in tests).
- `risk_parity_objective(w, Cov)` -- the real **relative** formulation
  `sum((contribution_i / target - 1)^2)` where `target = variance / n`
  -- deliberately not the naive absolute-difference formulation, which
  is documented as the actual root cause of the historical
  `systematic-equity-research` risk-parity bug this module's own
  docstring references.
- `is_risk_parity_converged(w, Cov, relative_tolerance)` -- a real,
  independent post-solve check: every security's own risk contribution
  must be within `relative_tolerance` of the equal-contribution target.

## Stage 5 -- optimizer (`optimization/optimizer.py`)

`optimize_portfolio(opt_input, objective, constraints, optimizer_cfg,
portfolio_id, cost_model=None)`:

1. Builds real `scipy` bounds from any `MaxPositionWeightConstraint`
   (tightest wins if more than one), and real `scipy` inequality
   constraint functions, one per `SectorExposureConstraint`/
   `FactorExposureConstraint`/`TurnoverConstraint`/`GrossLeverageConstraint`
   actually supplied, plus one real equality constraint
   `sum(w) == optimizer_cfg.target_net_exposure`.
2. Runs real `scipy.optimize.minimize` with `method="SLSQP"`.
3. **Independently reconstructs `converged`**, never reading
   `solver_result.success` directly:
   `converged = (no real constraint violations via check_all) AND
   (net exposure matches target within constraint_tolerance) AND
   (risk-parity convergence, if that was the objective)`. Any
   disagreement between `scipy`'s own flag and this real check is
   recorded as an explicit warning either direction.

When a real `cost_model` (Stage 7) is supplied with `objective ==
"mean_variance"`, its `estimate_cost(...)` is subtracted directly
inside the real `scipy_objective` closure -- the optimizer genuinely
trades expected return against real estimated cost, not applied only
after the fact. For `min_variance`/`risk_parity`, the cost model is
used only to populate `OptimizationResult.transaction_cost` for
reporting (those objectives have no expected-return term to trade
against).

## Stage 6 -- rebalancing (`rebalancing/rebalancer.py`)

`run_rebalancing_schedule(rebalance_dates, ...)` -- at each real
`rebalance_date` (processed in sorted order):

1. Builds the real universe/exposures as of that date only.
2. Estimates factor covariance and specific variance from
   `lookback_periods` real **trailing** dates only -- the core PIT rule
   this stage enforces: never uses any data dated after the current
   `rebalance_date`.
3. Builds the real stock covariance (Stage 2) and calls
   `optimize_portfolio` (Stage 5).
4. Carries `result.target_weights` forward as the next iteration's
   `current_weights` -- so turnover and transaction cost at the next
   rebalance are computed against the real prior decision, not re-zeroed.

`RebalancingSchedule(results, warnings_by_date)` -- the real ordered
collection of every `OptimizationResult` produced plus any
date-specific warnings.

## Stage 7 -- transaction costs (`backtest/transaction_costs.py`)

`TransactionCostModel(linear_cost_bps=10.0, market_impact_coefficient=0.0)`
(both RESEARCH ASSUMPTION -- see
[`research_assumptions.md`](research_assumptions.md)).

`estimate_cost(target_weights, current_weights, security_ids)` -- the
real formula, summed across `security_ids`:

```
trade_size = |target_weight - current_weight|   # absent key => 0.0
cost += trade_size * (linear_cost_bps / 10_000)
cost += market_impact_coefficient * trade_size**1.5   # only if > 0
```

A documented simplification: no real bid-ask spread, no venue-specific
modeling, no participation-rate throttling -- a genuine linear-plus-
optional-impact cost, not a full microstructure model.

## Stage 8 -- backtest (`backtest/backtest.py`)

`run_backtest(results, market_observations)` -- holds each rebalance's
real `target_weights` from its own `as_of` to the next rebalance's
`as_of`, computes the real weighted period return via
`_period_realized_returns` (reused directly from `factor-model-engine`'s
own market data), then:

```
gross_returns[t]       = real weighted portfolio return over period t
transaction_costs[t]   = results[t].transaction_cost
net_returns[t]          = gross_returns[t] - transaction_costs[t]
cumulative_net_return   = real compounded product of (1 + net_returns)
```

`BacktestResult` then calls `alpha-validation-toolkit`'s own real
`quant_toolkit.metrics.sharpe_ratio` and `probabilistic_sharpe_ratio`
directly on the real `net_returns` series -- this is genuine, verified
reuse of that shared toolkit (not a stub, not a reimplementation), and
is the one real place in this engine where that dependency is actually
exercised.

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

## Next up: Stage 5 -- Optimizer

A QP solver wrapper (`scipy.optimize`) that actually calls
`constraints.check_all` after solving -- real convergence, not a
blind pass-through of the solver's own `success` flag.

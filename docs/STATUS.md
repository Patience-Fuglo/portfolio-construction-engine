# Build status

Tracks stage-by-stage completion, mirroring `factor-model-engine`'s own
`docs/STATUS.md` convention, so work is resumable across sessions.

## Stage 1 -- Foundation: COMPLETE

`OptimizationInput`/`OptimizationResult` schemas (`data/schemas.py`),
`EngineConfig`/`load_config` (`config.py`), dependency wiring to
`factor-model-engine` (pinned git dependency, confirmed resolving
cleanly in a fresh venv) and `alpha-validation-toolkit`. 8 tests
passing. Pushed.

## Next up: Stage 2 -- Risk Model

Build the real stock-level covariance matrix from factor-model-
engine's own `FactorCovarianceResult` + specific-variance outputs:
`Cov = B·Σ_f·Bᵗ + diag(specific_variance)`, where `B` is the real
stock x factor exposure matrix (factor-model-engine's
`FactorExposureRow`s, pivoted).

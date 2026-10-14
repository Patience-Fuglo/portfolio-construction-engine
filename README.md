# portfolio-construction-engine

A point-in-time portfolio construction system: a stock-level risk
model built from `factor-model-engine`'s own factor covariance and
specific-risk outputs, real composable constraints, a mean-variance/
risk-parity optimizer with genuine post-solve convergence
verification, turnover-aware rebalancing, transaction costs, and a
backtesting harness.

**Stages 1–7 of 10 are complete**: schemas, config, dependency wiring
to `factor-model-engine` and `alpha-validation-toolkit`, a real
stock-level risk model, five composable real constraint types, real
mean-variance/min-variance/risk-parity objectives, a `scipy.optimize`-
based solver that never trusts the solver's own `success` flag,
turnover-aware rebalancing across a real date range that re-estimates
the whole risk model at each date using only trailing (never future)
history, and a cost-aware mean-variance objective that genuinely
trades off expected alpha against a real, simple transaction-cost
model. See [`docs/STATUS.md`](docs/STATUS.md) for stage-by-stage
build status.

## Relationship to this portfolio's other repos

Depends on (pinned git dependencies):
- [`alpha-validation-toolkit`](https://github.com/Patience-Fuglo/alpha-validation-toolkit)
  for shared statistical tooling (Sharpe/PSR/DSR for backtest
  evaluation).
- [`factor-model-engine`](https://github.com/Patience-Fuglo/factor-model-engine)
  for its point-in-time schemas and, critically, its factor covariance
  and specific-risk outputs -- this repo's stock-level covariance
  matrix is built from them (`Cov = B·Σ_f·Bᵗ + diag(specific_variance)`),
  never re-estimated independently.

Does not modify or replace any other repo in this portfolio.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
python -m pytest tests/ -v
```

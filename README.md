# portfolio-construction-engine

A point-in-time portfolio construction system: a stock-level risk
model built from `factor-model-engine`'s own factor covariance and
specific-risk outputs, real composable constraints, a mean-variance/
risk-parity optimizer with genuine post-solve convergence
verification, turnover-aware rebalancing, transaction costs, and a
backtesting harness.

**All 10 stages are complete.** Schemas, config, dependency wiring to
`factor-model-engine` and `alpha-validation-toolkit`; a real
stock-level risk model built from factor-model-engine's own factor
covariance and specific-risk outputs; five composable real constraint
types; real mean-variance/min-variance/risk-parity objectives; a
`scipy.optimize`-based solver that never trusts the solver's own
`success` flag; turnover-aware rebalancing across a real date range,
point-in-time safe; a cost-aware mean-variance objective and a real,
simple transaction-cost model; a backtest engine that forward-applies
a real schedule and measures realized P&L net of real costs, reusing
`alpha-validation-toolkit`'s Sharpe/PSR; one reproducible JSON/
Markdown report; and a comprehensive point-in-time check plus a real
CLI-chain subprocess test. See [`docs/STATUS.md`](docs/STATUS.md) for
the full stage-by-stage build history, the final architecture/formula
summary, test evidence, and an honest list of limitations --
including a real, investigated floating-point non-determinism finding
in the underlying SLSQP solver, caught by Stage 10's own hardening
pass and documented rather than hidden.

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
python scripts/run_portfolio_construction.py --n-periods-months 36 --objective min_variance --linear-cost-bps 10
```

## Repository layout

```
src/portfolio_construction/
  data/schemas.py              # OptimizationInput, OptimizationResult
  risk_model/stock_covariance.py  # build_exposure_matrix(),
                                   # build_stock_covariance_matrix()
  constraints/constraints.py       # five composable constraint types + check_all()
  optimization/
    objectives.py                   # mean_variance/min_variance/risk_parity
    optimizer.py                     # optimize_portfolio() -- real post-solve verification
  rebalancing/rebalancer.py           # run_rebalancing_schedule()
  backtest/
    transaction_costs.py              # TransactionCostModel
    backtest.py                        # run_backtest()
  reporting/report.py                  # report_to_dict()/render_markdown_report()

tests/
  unit/           # schemas, risk model, constraints, objectives, optimizer, transaction costs, backtest
  integration/     # cross-repo (factor-model-engine) integration, rebalancing, reporting,
                    # the comprehensive point-in-time check, real CLI-chain subprocess test

scripts/
  run_portfolio_construction.py   # end-to-end: synthetic data -> schedule -> backtest -> report
```

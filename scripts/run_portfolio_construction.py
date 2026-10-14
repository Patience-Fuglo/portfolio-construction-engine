#!/usr/bin/env python3
"""End-to-end CLI: generates factor-model-engine's own synthetic
dataset in-process, runs a real quarterly rebalancing schedule against
it, backtests the result net of a real transaction-cost model, and
writes one reproducible JSON + Markdown report.

Usage:
    python scripts/run_portfolio_construction.py --n-periods-months 36 --objective min_variance [--linear-cost-bps 10.0] [--max-position-weight 0.2] [--output-dir outputs/reports]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from factor_model.config import EngineConfig as FactorModelConfig, UniverseConfig
from factor_model.data import SyntheticDataConfig, fundamental_observations_from_frame, generate_synthetic_dataset, market_observations_from_frame

from portfolio_construction.backtest import TransactionCostModel, run_backtest
from portfolio_construction.config import EngineConfig
from portfolio_construction.constraints import GrossLeverageConstraint, MaxPositionWeightConstraint
from portfolio_construction.rebalancing import run_rebalancing_schedule
from portfolio_construction.reporting import render_markdown_report, report_to_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-securities", type=int, default=25)
    parser.add_argument("--n-periods-months", type=int, default=36)
    parser.add_argument("--lookback-periods", type=int, default=12)
    parser.add_argument("--objective", default="min_variance", choices=["mean_variance", "min_variance", "risk_parity"])
    parser.add_argument("--max-position-weight", type=float, default=0.2)
    parser.add_argument("--linear-cost-bps", type=float, default=0.0)
    parser.add_argument("--portfolio-id", default="portfolio")
    parser.add_argument("--output-dir", default="outputs/reports")
    args = parser.parse_args()

    dataset = generate_synthetic_dataset(SyntheticDataConfig(n_securities=args.n_securities, n_sectors=5, n_periods_months=args.n_periods_months))
    market_obs = market_observations_from_frame(dataset.market)
    fundamental_obs = fundamental_observations_from_frame(dataset.fundamentals)
    dates = sorted(dataset.market["as_of"].unique())
    rebalance_dates = [pd.Timestamp(d) for d in dates[args.lookback_periods::3]]

    fm_cfg = FactorModelConfig(universe=UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0))
    pc_cfg = EngineConfig()
    cost_model = TransactionCostModel(linear_cost_bps=args.linear_cost_bps) if args.linear_cost_bps > 0 else None

    def expected_returns_source(as_of, security_ids):
        return {sid: 0.0 for sid in security_ids}

    def constraints_factory(as_of, exposure_matrix):
        return [MaxPositionWeightConstraint(max_weight=args.max_position_weight, long_only=True), GrossLeverageConstraint(max_gross_leverage=1.0)]

    schedule = run_rebalancing_schedule(
        market_obs, fundamental_obs, rebalance_dates, args.lookback_periods, args.objective,
        expected_returns_source, constraints_factory, fm_cfg, pc_cfg, args.portfolio_id, cost_model=cost_model,
    )
    backtest = run_backtest(schedule.results, market_obs)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"report_{args.portfolio_id}.json"
    md_path = out_dir / f"report_{args.portfolio_id}.md"

    payload = report_to_dict(schedule, backtest, args.portfolio_id)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(md_path, "w") as f:
        f.write(render_markdown_report(schedule, backtest, args.portfolio_id))

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"{len(schedule.results)} rebalances, all converged: {payload['all_converged']}")
    print(f"cumulative net return: {backtest.cumulative_net_return:.4%}")


if __name__ == "__main__":
    main()

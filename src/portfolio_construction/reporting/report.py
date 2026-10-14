"""Stage 9 -- renders a real ``RebalancingSchedule`` (Stage 6) and its
``BacktestResult`` (Stage 8) into one reproducible report: a
JSON-serializable dict, and a professional Markdown document. Both
derived from the SAME two real objects -- never recomputed separately,
so they can't silently disagree with each other. Mirrors
`factor-model-engine`'s own Stage 9 reporting convention.
"""

from __future__ import annotations

from ..backtest import BacktestResult
from ..rebalancing import RebalancingSchedule


def report_to_dict(schedule: RebalancingSchedule, backtest: BacktestResult, portfolio_id: str) -> dict:
    results = schedule.results
    return {
        "portfolio_id": portfolio_id,
        "n_rebalances": len(results),
        "rebalance_dates": [str(r.as_of.date()) for r in results],
        "all_converged": all(r.converged for r in results) if results else False,
        "final_weights": results[-1].target_weights if results else {},
        "total_transaction_cost": sum(r.transaction_cost for r in results),
        "backtest": backtest.to_dict(),
        "warnings_by_date": {str(d.date()): w for d, w in schedule.warnings_by_date.items() if w},
    }


def render_markdown_report(schedule: RebalancingSchedule, backtest: BacktestResult, portfolio_id: str) -> str:
    payload = report_to_dict(schedule, backtest, portfolio_id)
    lines: list[str] = []
    lines.append("# Portfolio Construction Engine -- Research Report")
    lines.append("")
    lines.append(f"**Portfolio:** `{payload['portfolio_id']}`  **Rebalances:** {payload['n_rebalances']}  "
                 f"**All converged:** {payload['all_converged']}")
    lines.append("")
    lines.append("All data in this report is synthetic and non-proprietary.")
    lines.append("")

    lines.append("## Rebalancing Schedule")
    lines.append(f"- Dates: {', '.join(payload['rebalance_dates'])}")
    lines.append(f"- Total transaction cost across the schedule: {payload['total_transaction_cost']:.4%}")
    lines.append("")

    lines.append("## Final Target Weights")
    lines.append("| Security | Weight |")
    lines.append("|---|---|")
    for security_id, weight in sorted(payload["final_weights"].items()):
        lines.append(f"| {security_id} | {weight:.4f} |")
    lines.append("")

    backtest_payload = payload["backtest"]
    lines.append("## Backtest")
    lines.append(f"- Cumulative net return: {backtest_payload['cumulative_net_return']:.4%}")
    lines.append(f"- Sharpe ratio (per-period): {backtest_payload['sharpe_ratio']:.4f}")
    lines.append(f"- Probabilistic Sharpe ratio: {backtest_payload['probabilistic_sharpe_ratio']:.4f}")
    lines.append(f"- Periods: {backtest_payload['n_periods']}")
    lines.append("")

    if payload["warnings_by_date"]:
        lines.append("## Warnings by Rebalance Date")
        for date, warnings in sorted(payload["warnings_by_date"].items()):
            for w in warnings:
                lines.append(f"- {date}: {w}")
        lines.append("")

    return "\n".join(lines)

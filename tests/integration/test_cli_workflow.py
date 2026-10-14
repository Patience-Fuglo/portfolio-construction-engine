"""Stage 10 -- runs this repo's real CLI script as a subprocess,
exactly as a researcher would from the command line, confirming the
documented README workflow actually works end to end.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.slow
def test_run_portfolio_construction_cli_produces_a_real_report(tmp_path):
    output_dir = tmp_path / "reports"
    result = subprocess.run(
        [sys.executable, "scripts/run_portfolio_construction.py", "--n-periods-months", "30",
         "--objective", "min_variance", "--linear-cost-bps", "10", "--output-dir", str(output_dir)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert (output_dir / "report_portfolio.json").exists()
    assert (output_dir / "report_portfolio.md").exists()
    assert "cumulative net return" in result.stdout.lower()

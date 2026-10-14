"""Stage 2 -- builds a real stock-level covariance matrix from
factor-model-engine's own outputs, via the classic (Barra-style)
decomposition:

    Cov = B . Sigma_f . B^T + diag(specific_variance)

where ``B`` is the real stock x factor STANDARDIZED exposure matrix
(factor-model-engine's ``FactorExposureRow``s, pivoted), ``Sigma_f`` is
the real factor covariance matrix (``factor_model.risk.
FactorCovarianceResult.covariance``), and ``specific_variance`` is each
stock's own real idiosyncratic-return variance
(``factor_model.risk.compute_specific_variance``). This repo never
re-estimates any of these three real inputs independently -- it only
combines them.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from factor_model.data.schemas import FactorExposureRow
from factor_model.factors import factor_exposure_rows_to_frame

from ..config import RiskModelConfig


def build_exposure_matrix(stock_exposure_rows: Sequence[FactorExposureRow]) -> pd.DataFrame:
    """The real stock x factor STANDARDIZED exposure matrix (``B``),
    pivoted from factor-model-engine's own stock-level
    ``FactorExposureRow``s at one ``as_of``.
    """
    frame = factor_exposure_rows_to_frame(stock_exposure_rows)
    return frame.pivot(index="security_id", columns="factor_name", values="standardized_score")


def build_stock_covariance_matrix(
    exposure_matrix: pd.DataFrame,
    factor_covariance: pd.DataFrame,
    specific_variance: pd.Series,
    risk_cfg: RiskModelConfig,
) -> tuple[pd.DataFrame, list[str]]:
    """Real matrix construction, two explicit, documented (never
    silent) assumptions where a real input is missing:

    1. A security missing a real exposure reading for a given factor
       contributes exactly 0 (neutral, since standardized exposures are
       zero-mean) to that factor's term -- warned, not silently assumed.
    2. A security with no real specific-variance estimate (e.g. too
       little real history for factor-model-engine's own Stage 6 to
       estimate it) is floored at ``risk_cfg.min_specific_variance`` --
       never fabricated as exactly 0 idiosyncratic risk, which no real
       factor model can honestly claim for any real security.
    """
    security_ids = exposure_matrix.index.tolist()
    factors = factor_covariance.columns.tolist()
    warnings: list[str] = []

    aligned_exposure = exposure_matrix.reindex(columns=factors)
    n_missing_cells = int(aligned_exposure.isna().sum().sum())
    if n_missing_cells:
        warnings.append(
            f"{n_missing_cells} real security/factor exposure cells were missing and treated as "
            "zero (neutral) exposure for covariance construction"
        )
    exposure_values = aligned_exposure.fillna(0.0).to_numpy()
    factor_cov_term = exposure_values @ factor_covariance.to_numpy() @ exposure_values.T

    specific_var_aligned = specific_variance.reindex(security_ids)
    missing_specific = specific_var_aligned[specific_var_aligned.isna()].index.tolist()
    if missing_specific:
        warnings.append(
            f"{len(missing_specific)} real securities have no specific-variance estimate -- floored "
            f"at {risk_cfg.min_specific_variance}, never fabricated as zero idiosyncratic risk: {sorted(missing_specific)}"
        )
    specific_var_filled = specific_var_aligned.fillna(risk_cfg.min_specific_variance).clip(lower=risk_cfg.min_specific_variance)

    covariance = factor_cov_term + np.diag(specific_var_filled.to_numpy())
    covariance_df = pd.DataFrame(covariance, index=security_ids, columns=security_ids)
    return covariance_df, warnings

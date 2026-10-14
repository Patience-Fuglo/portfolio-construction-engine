import numpy as np
import pandas as pd
import pytest
from factor_model.data.schemas import ExposureSubject, FactorExposureRow

from portfolio_construction.config import RiskModelConfig
from portfolio_construction.risk_model import build_exposure_matrix, build_stock_covariance_matrix


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _exposure(security_id: str, factor_name: str, value: float) -> FactorExposureRow:
    return FactorExposureRow(
        subject_type=ExposureSubject.STOCK, subject_id=security_id, as_of=_ts("2020-06-30"),
        factor_name=factor_name, exposure_value=value, source_known_as_of=_ts("2020-06-30"),
    )


def test_build_exposure_matrix_pivots_correctly():
    rows = [_exposure("A", "size", 0.5), _exposure("A", "momentum", -0.2), _exposure("B", "size", -0.1), _exposure("B", "momentum", 0.3)]
    matrix = build_exposure_matrix(rows)
    assert matrix.loc["A", "size"] == pytest.approx(0.5)
    assert matrix.loc["B", "momentum"] == pytest.approx(0.3)


def test_stock_covariance_matches_the_real_matrix_identity():
    exposure_matrix = pd.DataFrame({"size": [1.0, -1.0], "momentum": [0.5, 0.5]}, index=["A", "B"])
    factor_covariance = pd.DataFrame({"size": [0.02, 0.0], "momentum": [0.0, 0.01]}, index=["size", "momentum"])
    specific_variance = pd.Series({"A": 0.05, "B": 0.03})

    covariance, warnings = build_stock_covariance_matrix(exposure_matrix, factor_covariance, specific_variance, RiskModelConfig())
    B = exposure_matrix.to_numpy()
    expected = B @ factor_covariance.to_numpy() @ B.T + np.diag([0.05, 0.03])
    np.testing.assert_allclose(covariance.to_numpy(), expected)
    assert warnings == []


def test_stock_covariance_is_symmetric_and_positive_semidefinite():
    rng = np.random.default_rng(0)
    n_securities, n_factors = 15, 4
    exposure_matrix = pd.DataFrame(
        rng.normal(size=(n_securities, n_factors)),
        index=[f"S{i}" for i in range(n_securities)], columns=[f"F{i}" for i in range(n_factors)],
    )
    raw = rng.normal(size=(n_factors, n_factors))
    factor_covariance = pd.DataFrame(raw @ raw.T, index=exposure_matrix.columns, columns=exposure_matrix.columns)
    specific_variance = pd.Series(rng.uniform(0.01, 0.1, size=n_securities), index=exposure_matrix.index)

    covariance, _ = build_stock_covariance_matrix(exposure_matrix, factor_covariance, specific_variance, RiskModelConfig())
    np.testing.assert_allclose(covariance.to_numpy(), covariance.to_numpy().T)
    eigenvalues = np.linalg.eigvalsh(covariance.to_numpy())
    assert (eigenvalues >= -1e-8).all()


def test_stock_covariance_floors_missing_specific_variance_never_zero():
    exposure_matrix = pd.DataFrame({"size": [1.0]}, index=["A"])
    factor_covariance = pd.DataFrame({"size": [0.02]}, index=["size"])
    specific_variance = pd.Series(dtype=float)  # no real estimate for A

    cfg = RiskModelConfig(min_specific_variance=0.001)
    covariance, warnings = build_stock_covariance_matrix(exposure_matrix, factor_covariance, specific_variance, cfg)
    diag_specific = covariance.loc["A", "A"] - (1.0 * 0.02 * 1.0)
    assert diag_specific == pytest.approx(0.001)
    assert any("no specific-variance estimate" in w for w in warnings)


def test_stock_covariance_treats_missing_factor_exposure_as_neutral_with_warning():
    exposure_matrix = pd.DataFrame({"size": [1.0]}, index=["A"])  # missing "momentum" entirely
    factor_covariance = pd.DataFrame({"size": [0.02, 0.0], "momentum": [0.0, 0.01]}, index=["size", "momentum"])
    specific_variance = pd.Series({"A": 0.05})

    covariance, warnings = build_stock_covariance_matrix(exposure_matrix, factor_covariance, specific_variance, RiskModelConfig())
    assert any("neutral" in w for w in warnings)
    # momentum exposure treated as 0 -> only the size term + specific variance contribute
    assert covariance.loc["A", "A"] == pytest.approx(1.0 * 0.02 * 1.0 + 0.05)

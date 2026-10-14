"""Engine configuration: every ambiguous research decision lives here,
in YAML, with a recommended default and reasoning in
``docs/research_assumptions.md`` -- never hardcoded in source. Same
plain-``@dataclass`` house style as ``factor-model-engine``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass(frozen=True)
class RunConfig:
    output_dir: str = "outputs"
    random_seed: int = 1337


@dataclass(frozen=True)
class RiskModelConfig:
    # RESEARCH ASSUMPTION -- a floor added to every security's specific
    # variance before it enters the stock covariance matrix, so a
    # security with too little real history for Stage-6-of-factor-
    # model-engine's specific-variance estimate never silently
    # contributes zero idiosyncratic risk to the optimizer
    min_specific_variance: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.min_specific_variance < 0:
            raise ValueError("min_specific_variance cannot be negative")


@dataclass(frozen=True)
class OptimizerConfig:
    # RESEARCH ASSUMPTION -- mean-variance objective's risk-aversion
    # coefficient (maximize expected_return - risk_aversion * variance)
    risk_aversion: float = 5.0
    # RESEARCH ASSUMPTION -- how far a constraint's own check may be
    # violated before the optimizer's post-solve verification flags
    # non-convergence, rather than trusting the solver's own success flag
    constraint_tolerance: float = 1.0e-6
    # RESEARCH ASSUMPTION -- sum(weights) target: 1.0 = fully-invested
    # long-only (the default convention); 0.0 would be a dollar-neutral
    # long/short book instead
    target_net_exposure: float = 1.0
    # RESEARCH ASSUMPTION -- relative tolerance for is_risk_parity_converged
    risk_parity_tolerance: float = 0.05
    # RESEARCH ASSUMPTION -- scipy.optimize.minimize's own max iterations
    max_iterations: int = 500

    def __post_init__(self) -> None:
        if self.risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive")
        if self.constraint_tolerance <= 0:
            raise ValueError("constraint_tolerance must be positive")
        if self.risk_parity_tolerance <= 0:
            raise ValueError("risk_parity_tolerance must be positive")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")


@dataclass(frozen=True)
class EngineConfig:
    run: RunConfig = field(default_factory=RunConfig)
    risk_model: RiskModelConfig = field(default_factory=RiskModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


def load_config(path: str) -> EngineConfig:
    """Load a real YAML config file into a validated EngineConfig.
    Missing sections fall back to each dataclass's own documented
    default -- never a silent empty/zero value."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    run = RunConfig(**raw.get("run", {}))
    risk_model = RiskModelConfig(**raw.get("risk_model", {}))
    optimizer = OptimizerConfig(**raw.get("optimizer", {}))

    return EngineConfig(run=run, risk_model=risk_model, optimizer=optimizer)

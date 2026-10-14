# Research assumptions

Every field below is a real modeling judgment call, not a safe engineering
default, and needs the researcher's explicit sign-off before being relied on
for any real analysis. Each lives in `configs/default.yaml`, tagged
`# RESEARCH ASSUMPTION` at the point of definition, and in
`src/portfolio_construction/config.py`'s own inline comments -- this
document consolidates both into one reference, matching
`factor-model-engine`'s own convention.

## Risk model (`risk_model:` in `configs/default.yaml`)

**`min_specific_variance: 1.0e-6`**
A floor added to every security's specific (idiosyncratic) variance before
it enters the stock covariance matrix (`Cov = B·Σ_f·Bᵗ +
diag(specific_variance)`, `risk_model/stock_covariance.py`). A security
with too little real history for `factor-model-engine`'s own Stage 6
specific-variance estimate never silently contributes exactly zero
idiosyncratic risk to the optimizer -- no real factor model can honestly
claim a security has zero idiosyncratic risk, so a missing estimate is
floored, never fabricated as 0.0.

## Optimizer (`optimizer:`)

**`risk_aversion: 5.0`**
The mean-variance objective's risk-aversion coefficient: the optimizer
maximizes `expected_return - risk_aversion * variance`. A round,
moderate default -- a specific real mandate's actual risk tolerance should
override this.

**`constraint_tolerance: 1.0e-6`**
How far a constraint's own `check()` may be violated before the
optimizer's independent post-solve verification flags non-convergence,
rather than trusting `scipy.optimize`'s own `success` flag (a real,
directly observed failure mode -- see `docs/STATUS.md` Stage 5). A tight
numerical tolerance, not a real economic leeway.

**`target_net_exposure: 1.0`**
The equality constraint `sum(weights) == target_net_exposure` the
optimizer always enforces. `1.0` is the real default convention for a
fully-invested, long-only book; `0.0` would instead describe a
dollar-neutral long/short book. This is a real mandate choice, not a
technical default.

**`risk_parity_tolerance: 0.05`**
The relative tolerance `is_risk_parity_converged` uses to independently
verify that every asset's real risk contribution is genuinely within 5%
of equal, before trusting a risk-parity solve as converged -- never
inferred from the solver's own `success` flag or the objective value
alone (both have been real, directly observed to be misleading; see the
systematic-equity-research risk-parity bug referenced throughout this
repo's own docstrings).

**`max_iterations: 500`**
`scipy.optimize.minimize`'s own real iteration budget -- a safe
engineering ceiling for the problem sizes this engine targets, not a
research judgment about convergence behavior itself.

## Not a research assumption (real engineering defaults)

`run.output_dir`, `run.random_seed` are safe engineering defaults, not
modeling choices, and are not listed here.

## Reused real inputs, not re-estimated here

This engine makes no independent research assumptions about factor
construction, universe eligibility, or factor-return estimation -- those
real judgment calls live in `factor-model-engine`'s own
`docs/research_assumptions.md` and are consumed here as given (see
`docs/data_dictionary.md` for exactly which real objects cross that
boundary).

# Supplementary Appendix — Extended Methods

**Dissonance Field Synthesis in Mineralocorticoid-Receptor-Antagonist Trials
for HFpEF: Full Mathematical Specification**

Supplement to: Ahmad M et al. *Circulation* / *JAMA Cardiology* (under review).

---

## S1. Formal Definition of Dissonance Field Synthesis

### S1.1 Boundary conditions

Let $\mathcal{T} = \{T_1, \ldots, T_K\}$ be a set of $K$ clinical trials. Each
trial $T_i$ contributes a **boundary-condition record** consisting of:

- An anchor covariate vector $\mathbf{x}_i \in \mathbb{R}^D$, representing the
  trial's population at a summary level ($D = 7$ in this application).
- An observed outcome $y_i = \hat{\theta}_i$ (log-hazard ratio for the primary
  composite endpoint) with standard error $\sigma_i$.
- Covariate range bounds $[\mathbf{x}_i^{\min}, \mathbf{x}_i^{\max}]$ describing
  the support of the trial's population.
- A set of secondary outcomes (cause-specific mortality, hospitalisation) and
  safety outcomes (ΔK⁺, ΔSBP, ΔeGFR) with their own estimates and standard errors.
- Design priors: placebo event rate, loss-to-follow-up fraction, adherence proxy.

The boundary-condition record schema is formally defined in `dfs/schema.py`. The
JSON format for each trial in this application is stored in `data/mra_hfpef/`.

### S1.2 Pairwise dissonance

For each ordered pair $(i, j)$ with $i < j$, the **observed dissonance** is:

$$d_{ij} = \frac{|\hat{\theta}_i - \hat{\theta}_j|}{\sqrt{\sigma_i^2 + \sigma_j^2}}$$

This is a z-score for the hypothesis that $\theta_i = \theta_j$; values above 2
indicate disagreement at the 5% level under the assumption of independent
normally-distributed estimates. No pooling is performed at this stage.

The **covariate delta** for each dimension $k$ is:

$$\Delta x_{ij}^{(k)} = x_i^{(k)} - x_j^{(k)}$$

where $x_i^{(k)}$ is the $k$-th component of the anchor covariate vector for
trial $i$. These deltas form the explanatory variables for the dissonance
decomposition.

### S1.3 The GP field

Define the **effect field** $f : \mathbb{R}^D \to \mathbb{R}$ as the log-hazard
ratio of the primary composite endpoint as a function of covariate vector
$\mathbf{x}$. The field is modelled as a Gaussian process:

$$f(\mathbf{x}) \sim \mathcal{GP}\bigl(m(\mathbf{x}),\; k(\mathbf{x}, \mathbf{x}')\bigr)$$

**Prior mean:**

$$m(\mathbf{x}) = \beta_0 + \beta_{\text{occ}} \cdot x^{(\text{occ})}$$

where $x^{(\text{occ})}$ is the MR-occupancy component of $\mathbf{x}$,
$\beta_0 = 0$ (null prior on baseline effect at zero occupancy), and
$\beta_{\text{occ}} \in [-0.15, -0.05]$ (mild negative: occupancy of 1.0 shifts
the log-HR prior by approximately 5–15%, weak enough that trial data dominates
with $K \geq 5$).

**Kernel:**

$$k(\mathbf{x}, \mathbf{x}') = \sigma^2 \cdot M_{5/2}\!\left(\|\mathbf{x} - \mathbf{x}'\|_L\right)$$

where:

$$M_{5/2}(r) = \left(1 + \sqrt{5}\,r + \frac{5}{3}r^2\right) \exp\!\left(-\sqrt{5}\,r\right)$$

$$\|\mathbf{x} - \mathbf{x}'\|_L = \sqrt{\sum_{k=1}^D \frac{(x^{(k)} - x'^{(k)})^2}{\ell_k^2}}$$

and $L = \text{diag}(\ell_1, \ldots, \ell_D)$ is a diagonal matrix of per-covariate
ARD length-scales. The Matérn 5/2 kernel is twice continuously differentiable,
making it suitable for GP models of physical effects that are smooth but not
infinitely so. The ARD parameterisation allows each covariate to have an
independent effective resolution.

**Likelihood:**

$$y_i \mid f(\mathbf{x}_i) \sim \mathcal{N}\!\left(f(\mathbf{x}_i),\; \sigma_i^2\right)$$

with heteroscedastic noise variance $v_i = \sigma_i^2$ fixed at the squared
standard error of each trial's log-HR estimate.

**Unconstrained posterior:** Standard Gaussian process closed form. Let
$\mathbf{X} = [\mathbf{x}_1, \ldots, \mathbf{x}_K]^T$,
$\mathbf{y} = [y_1, \ldots, y_K]^T$,
$K_{XX} = k(\mathbf{X}, \mathbf{X})$,
$K_{X*} = k(\mathbf{X}, \mathbf{x}_*)$ for a test point $\mathbf{x}_*$,
and $\Lambda = \text{diag}(v_1, \ldots, v_K)$. Then:

$$\mu_*^{\text{unc}} = m_* + K_{X*}^T \left(K_{XX} + \Lambda\right)^{-1} (\mathbf{y} - \mathbf{m})$$

$$\sigma_*^{2,\text{unc}} = k_{**} - K_{X*}^T \left(K_{XX} + \Lambda\right)^{-1} K_{X*}$$

where $m_* = m(\mathbf{x}_*)$ and $k_{**} = k(\mathbf{x}_*, \mathbf{x}_*)$.

---

## S2. Type-II Marginal Likelihood (ML-II) Hyperparameter Optimisation

### S2.1 Objective

The hyperparameters $\boldsymbol{\psi} = (\sigma^2, \ell_1, \ldots, \ell_D)$ are
fitted by maximising the log marginal likelihood (evidence):

$$\log p(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\psi}) = -\frac{1}{2}\mathbf{y}^T C^{-1} \mathbf{y} - \frac{1}{2}\log|C| - \frac{K}{2}\log 2\pi$$

where $C = K_{XX} + \Lambda$. Equivalently, we minimise the negative log marginal
likelihood (NLML):

$$\mathcal{L}(\boldsymbol{\psi}) = \frac{1}{2}\mathbf{y}^T C^{-1} \mathbf{y} + \frac{1}{2}\log|C| + \frac{K}{2}\log 2\pi$$

### S2.2 Numerical implementation

Optimisation is performed over the log-parameterised space
$\boldsymbol{\phi} = (\log\sigma^2, \log\ell_1, \ldots, \log\ell_D)$ using
L-BFGS-B with bounds $\phi_k \in [-10, 10]$ for all $k$ (corresponding to
$\exp(-10) \approx 5 \times 10^{-5}$ to $\exp(10) \approx 22,000$ in the
original space). The covariance matrix $C$ is Cholesky-factorised for numerical
stability and for efficient computation of the log-determinant as
$\log|C| = 2 \sum_i \log L_{ii}$ where $L$ is the lower Cholesky factor.
A jitter term $10^{-8} I$ is added before factorisation.

Five random restarts are performed using independent draws from
$\phi_k^{(0)} \sim \mathcal{N}(\bar{\phi}_k, 1)$ where $\bar{\phi}_k$ is the
deterministic starting point, with RNG seed fixed for reproducibility. The
optimum across all restarts with successful convergence is retained.

All covariates are normalised to $[0, 1]$ before kernel evaluation: for each
dimension $k$, $\tilde{x}^{(k)} = (x^{(k)} - x_{\min}^{(k)}) /
(x_{\max}^{(k)} - x_{\min}^{(k)})$ where min and max are taken across training
trials.

### S2.3 ARD length-scale interpretation

A length-scale $\ell_k$ in normalised space measures the covariate separation
at which two trials have kernel correlation $\approx e^{-\sqrt{5}} \approx 0.15$.
A length-scale much smaller than the covariate spread implies the kernel assigns
near-zero covariance between trials that differ along dimension $k$; a length-scale
much larger than the covariate spread implies the kernel treats all trials as
essentially equivalent in that dimension.

In this application, the adherence-proxy spread across trials is approximately
0.50 normalised units (TOPCAT-Russia/Georgia at 0.40, FINEARTS-HF at 0.90,
normalised accordingly). A fitted length-scale of 0.44 is smaller than this
spread, meaning the GP assigns low covariance between trials with different
adherence-proxy values: adherence is the axis along which trial effects are
assumed to vary most rapidly.

---

## S3. Constrained GP via Virtual-Observation Projection

### S3.1 Hard inequality constraints

Following Da Veiga and Marrel [S1], hard inequality constraints are enforced by
augmenting the training set with virtual observations. Let $\{z_j\}_{j=1}^M$
be a set of $M$ virtual observation points sampled from the covariate support via
a Latin hypercube design. A hard inequality constraint $g(f(\mathbf{z}_j)) \geq 0$
is encoded by requiring the GP function value at each virtual point to satisfy
the constraint.

For the k_sign law, $g(f) = f$ (identity) applied to the ΔK⁺ safety GP
$f_{\Delta K} : \mathbb{R}^D \to \mathbb{R}$, with the constraint that
$f_{\Delta K}(\mathbf{z}) \geq 0$ for all $\mathbf{z}$ with
$z^{(\text{occ})} \in [0.1, 1.0]$.

### S3.2 Quadratic programme formulation

The constrained MAP of the GP posterior at the virtual points $\mathbf{f}_v =
[f(\mathbf{z}_1), \ldots, f(\mathbf{z}_M)]^T$ is found by solving the quadratic
programme:

$$\min_{\mathbf{f}_v} \;\; \frac{1}{2}\left(\mathbf{f}_v - \boldsymbol{\mu}_v^{\text{unc}}\right)^T \Sigma_v^{-1} \left(\mathbf{f}_v - \boldsymbol{\mu}_v^{\text{unc}}\right)$$

$$\text{subject to} \quad A \mathbf{f}_v \geq \mathbf{b}$$

where $\boldsymbol{\mu}_v^{\text{unc}}$ is the unconstrained GP posterior mean at
the virtual points, $\Sigma_v$ is the posterior covariance at the virtual points,
and $A\mathbf{f}_v \geq \mathbf{b}$ encodes the linear inequality constraints
($A = I$, $\mathbf{b} = \mathbf{0}$ for the k_sign law).

This programme is solved using CVXPY [S2] with the CLARABEL solver [S3]. The
constrained posterior at any test point $\mathbf{x}_*$ is then computed by
conditioning on both the trial observations $\mathbf{y}$ and the virtual
observations $\mathbf{f}_v^*$ obtained at the QP optimum.

### S3.3 Latin hypercube grid construction

Virtual observation points are sampled using a stratified Latin hypercube design.
For $M$ virtual points in $D$ dimensions, the hypercube divides each normalised
covariate axis into $M$ equal intervals; one point is sampled uniformly from each
interval, and the resulting $M \times D$ matrix is column-permuted independently
for each dimension. For the k_sign safety GP, $M = 50$ and the MR-occupancy
dimension is clamped to $[0.1, 1.0]$ to enforce the law only in the active
pharmacological region.

---

## S4. The Six Conservation Laws

Six conservation laws are defined in `dfs/conservation.py`. Their equations and
wiring status are given below.

### S4.1 Mortality decomposition (deferred to Phase-2)

$$\text{HR}_{\text{ACM}}(\mathbf{x}) = p(\mathbf{x}) \cdot \text{HR}_{\text{CV}}(\mathbf{x}) + q(\mathbf{x}) \cdot \text{HR}_{\text{non-CV}}(\mathbf{x})$$

where $p$ and $q = 1 - p$ are the trial-level baseline proportions of
cardiovascular and non-cardiovascular deaths respectively. This is an arithmetic
identity on the hazard-difference scale; violations indicate transcription error
or differential censoring. Classification: **hard**.

**Phase-2 blocker:** requires a multi-output GP where the ACM, CV-death, and
non-CV-death surfaces share a joint posterior; the constraint spans three separate
GP output vectors. An intrinsic co-regionalisation model or joint quadratic
programme over stacked output vectors is needed.

### S4.2 CV-death subdecomposition (deferred to Phase-2)

$$\text{HR}_{\text{CV}}(\mathbf{x}) = \sum_{c \in \mathcal{C}} \pi_c \cdot \text{HR}_c(\mathbf{x})$$

where $\mathcal{C} = \{\text{sudden death, pump failure, MI, stroke}\}$ and
$\pi_c$ are cause-specific proportions of CV death. Classification: **hard**.

**Phase-2 blocker:** same as S4.1.

### S4.3 Mechanism sign: potassium (k_sign) — wired in Phase-1b

$$\Delta K^+(\mathbf{x}) \geq 0 \quad \text{wherever} \quad x^{(\text{occ})} > 0$$

Aldosterone blockade conserves potassium by reducing aldosterone-mediated
excretion. A negative ΔK⁺ can occur only if the drug did not reach the receptor
(non-adherence) or if measurement timing captured a transient pre-equilibrium
state. Classification: **hard**.

**Implementation:** single-output GP on $\Delta K^+$ with hard inequality
enforced at $M = 50$ virtual points (§S3). Wired in Phase-1b B.

**Real-data result:** constraint non-binding; minimum posterior ΔK⁺ on virtual
grid = +0.047 mmol/L. All 6 trials satisfy the constraint.

### S4.4 Mechanism sign: blood pressure (sbp_sign) — deferred to Phase-2

$$\frac{\partial \Delta\text{SBP}}{\partial x^{(\text{occ})}}(\mathbf{x}) \leq 0$$

MR blockade reduces sodium retention, lowering SBP monotonically with occupancy.
Classification: **soft** (some normotensive subgroups show negligible SBP change;
the magnitude is small relative to measurement noise).

**Phase-2 blocker:** soft constraints require an additive penalty term in the QP
objective (quadratic slack or log-barrier), not a hard linear inequality. Current
`fit_constrained_gp` API enforces only hard inequalities.

### S4.5 Dose-response monotonicity (dose_monotonicity) — deferred to Phase-2

$$\frac{\partial f(\mathbf{x})}{\partial x^{(\text{occ})}} \leq 0 \quad \text{within each drug's tested dose range}$$

Within the tested dose range of each drug, increasing receptor occupancy is
expected to increase efficacy (more negative log-HR). Classification: **hard**.

**Phase-2 blocker:** enforcing a gradient constraint requires joint GP modelling
of $f$ and $\partial f / \partial x^{(\text{occ})}$ via the analytically derived
cross-covariance of the Matérn 5/2 kernel and its derivative. Alternatively,
finite-difference virtual observations on a fine dose grid approximate this
constraint, but require a sufficiently dense grid to avoid artefacts.

### S4.6 eGFR dip-plateau (egfr_dip_plateau) — deferred to Phase-2

$$\Delta\text{eGFR}_{\text{4 months}}(\mathbf{x}) < 0 \quad \text{and} \quad \Delta\text{eGFR}_{\text{24 months}}(\mathbf{x}) \geq \Delta\text{eGFR}_{\text{4 months}}(\mathbf{x})$$

MR blockade reduces glomerular hyperfiltration driven by aldosterone, causing an
acute eGFR dip followed by partial recovery as afferent arteriolar tone is
re-established. Classification: **soft** (timing of follow-up varies; the
constraint is approximate).

**Phase-2 blocker:** two extensions are needed — (i) soft QP penalty term (as in
S4.4), and (ii) a time-dependent constraint because the sign of ΔeGFR reverses
between 4 months and 24 months. This requires either a separate GP per timepoint
or a spatio-temporal kernel with a time axis added to the covariate space.

---

## S5. Mind-Change Price Map

### S5.1 Derivation

Let the current posterior at covariate point $\mathbf{x}$ be approximately
$\mathcal{N}(\mu, \sigma^2)$ where $\mu = \mu(\mathbf{x})$ and
$\sigma^2 = \sigma^2(\mathbf{x})$ are the GP posterior mean and variance. Define
an effective sample size $n_{\text{eff}} = v_{\text{trial}} / \sigma^2$ where
$v_{\text{trial}}$ is the per-patient log-HR variance for a hypothetical new
trial at the relevant endpoint (values in `dfs/decisions.py`).

A hypothetical new trial with $n$ patients, observed log-HR $T_{\text{obs}}$,
and per-patient variance $v_{\text{trial}}$ updates the posterior mean to:

$$\mu_{\text{new}} = \frac{\mu \cdot n_{\text{eff}} + T_{\text{obs}} \cdot n}{n_{\text{eff}} + n}$$

The **mind-change price** is the smallest $n$ such that
$\mu_{\text{new}} \geq T_{\text{cross}}$ (the prescribing threshold), i.e. the
smallest trial that would flip the recommendation. Setting $\mu_{\text{new}} = T_{\text{cross}}$
and solving:

$$\text{MCP}(\mathbf{x}) = n_{\text{eff}} \cdot \frac{\mu - T_{\text{cross}}}{T_{\text{cross}} - T_{\text{obs}}}$$

This is defined when $T_{\text{obs}} \neq T_{\text{cross}}$ and when
$\mu$ and $T_{\text{obs}}$ are on opposite sides of $T_{\text{cross}}$ (a new
trial that agrees with the current recommendation increases the price, not
decreases it). Implementation details and edge cases are in `dfs/mind_change.py`.

### S5.2 Prescribing thresholds

Decision thresholds are encoded in `dfs/decisions.py` (user-authored). For this
application:

| Endpoint | Recommend if log-HR below | Do not recommend if log-HR above |
|---|---|---|
| Primary composite | log(0.90) = −0.105 | log(1.00) = 0.000 |
| ACM | log(0.90) = −0.105 | log(1.00) = 0.000 |
| CV death | log(0.90) = −0.105 | log(1.00) = 0.000 |
| HF hospitalisation | log(0.85) = −0.163 | log(1.00) = 0.000 |

<!-- AUTHOR REVIEW: These thresholds reflect the defaults in decisions.py.
Mahmood should confirm they reflect current guideline-informed prescribing
practice. In particular, whether the HR < 0.90 recommendation threshold for
ACM and CV death aligns with current guideline thinking on what constitutes
clinically meaningful mortality reduction in HFpEF. -->

---

## S6. Boundary-Condition Record Schema

Each trial is stored as a JSON file conforming to the following schema (Python
dataclass defined in `dfs/schema.py`):

```
{
  "trial_id": str,
  "drug": str,
  "mr_occupancy_equivalent": float,        // 1.0 = spironolactone 25 mg/day
  "anchor_covariates": {
    "lvef": float,                          // %
    "egfr": float,                          // mL/min/1.73m²
    "age": float,                           // years
    "baseline_k": float,                    // mmol/L
    "dm_fraction": float,                   // proportion 0-1
    "mr_occupancy": float,                  // normalised occupancy
    "adherence_proxy": float               // score 0-1
  },
  "covariate_ranges": {                    // support within trial
    "<covariate>": [min, max], ...
  },
  "outcomes": {
    "<outcome_name>": {
      "log_hr": float,
      "se": float,
      "baseline_prop": float,              // fraction of composite
      "source": str                        // PMID or AACT ID
    }, ...
  },
  "safety": {
    "delta_k": {"value": float, "se": float, "source": str},
    "delta_sbp": {"value": float, "se": float, "source": str},
    "delta_egfr": {"value": float, "se": float, "source": str}
  },
  "design_priors": {
    "placebo_rate_per_yr": float,
    "ltfu_fraction": float,
    "adherence_proxy": float
  }
}
```

All outcome records include a `source` field documenting the provenance of the
value (PMID, AACT outcome identifier, or derivation rationale). Derived values
receive 2× standard-error inflation in the GP noise model.

---

## S7. Reproducibility Checklist

| Item | Value |
|---|---|
| Python version | 3.13.7 |
| NumPy version | ≥ 1.26 |
| SciPy version | ≥ 1.11 |
| CVXPY version | ≥ 1.5 |
| QP solver | CLARABEL (via CVXPY) |
| statsmodels version | ≥ 0.14 |
| matplotlib version | ≥ 3.8 |
| AACT snapshot date | 2026-04-12 |
| ML-II RNG seed (full pipeline) | 0 |
| ML-II RNG seed (LOO test) | 42 |
| Virtual grid size (k_sign) | 50 points, Latin hypercube |
| Full test suite | 83 passed, 1 skipped |
| Git commit | f230069 |
| Repository | https://github.com/mahmood726-cyber/dissonance-field-synthesis |

**To reproduce all results:**

```bash
git clone https://github.com/mahmood726-cyber/dissonance-field-synthesis
cd dissonance-field-synthesis
git checkout f230069
python -m pip install -e ".[dev]"
python -m pytest -v                                   # 83 passed, 1 skipped
python scripts/run_dfs.py \
    --manifest data/mra_hfpef/MANIFEST.json \
    --out outputs/
```

All artefacts in `outputs/` are deterministically regenerated from the above
command. The one intentional test skip is a combinatorial counting test that is
deferred to an integration-contract phase; it does not affect any reported result.

<!-- AUTHOR REVIEW: The LOO test ML-II hyperparameters (seed 42, 5 trials) differ
from the full-pipeline hyperparameters (seed 0, 6 trials): specifically, the
LOO adherence-proxy length-scale is 0.36 (reported in the LOO test output)
versus 0.44 for the full pipeline (from k_sign_constraint_report.json). Both
are well below 1.0 and confirm the same qualitative finding. The manuscript
reports the full-pipeline value (0.44) in the main text. The LOO value is
reported here for completeness. -->

---

## S8. Supplementary References

[S1] Da Veiga S, Marrel A. Gaussian process regression with linear inequality
constraints. *Reliab Eng Syst Saf* 2020;195:106732.

[S2] Diamond S, Boyd S. CVXPY: A Python-embedded modeling language for convex
optimization. *J Mach Learn Res* 2016;17(83):1–5.

[S3] Goulart PJ, Chen Y. Clarabel: An interior-point solver for conic programs
with quadratic objectives. arXiv:2212.09040.

<!-- AUTHOR REVIEW: Reference [S1] journal citation should be confirmed.
Reference [S3] may have a published venue by submission time — check arXiv page. -->

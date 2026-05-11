# Bayesian Model Selection — CxO Quick Reference

*How the M-W framework chooses the right equation from data*

---

## The Core Idea in One Sentence

Given competing explanations for the same data, pick the one with the highest posterior probability — penalising unnecessary complexity.

---

## The Fundamental Equation

**Posterior ∝ Likelihood × Prior**

| Term | Plain English | M-W Example |
|---|---|---|
| **Prior** | What we believed before seeing data | Classical physics structures: GM/c²a, Keplerian orbits |
| **Likelihood** | How well the model fits the observed data | R² = 0.9845 on Mercury perihelion measurements |
| **Posterior** | Updated belief after seeing data | The GR formula is now the most probable explanation |

---

## BIC — Bayesian Information Criterion

**BIC = n·log(σ²) + k·log(n)**

Where n = number of data points, k = number of free parameters, σ² = residual variance.

**Rule: lower BIC wins.**

| What it does | Why it matters |
|---|---|
| Measures fit to data | A model that fits poorly is penalised |
| Penalises extra parameters | A model with 10 parameters vs 2 must earn its complexity |
| Approximates the Bayes Factor | Selecting by BIC ≈ selecting by posterior probability |

**CxO translation:** BIC is Occam's Razor made mathematical. The simplest explanation that fits the data wins. In the GR rediscovery, 14 candidate structures competed — BIC selected k·GM/(c²·a·(1−e²)) as rank 1. Einstein's coefficient 6π then emerged from curve-fitting, not assumption.

---

## AIC — Akaike Information Criterion

**AIC = 2k − 2·log(L)**

Where k = parameters, L = maximum likelihood.

**Rule: lower AIC wins.**

| | BIC | AIC |
|---|---|---|
| Penalty for complexity | k·log(n) — grows with dataset size | 2k — fixed |
| Preferred when | Dataset is large (n > 100) | Dataset is small or prediction is the goal |
| Philosophy | Find the true model | Find the best predictive model |
| Used in M-W | Primary selector | Cross-check on small datasets |

**CxO translation:** AIC favours models that predict well. BIC favours models that are true. For physics discovery — where we want the actual law, not just a good fit — BIC is the right tool. For forecasting (revenue, churn), AIC often wins.

---

## Bayes Factor

**BF = P(data | Model A) / P(data | Model B)**

| BF value | Interpretation |
|---|---|
| > 100 | Decisive evidence for Model A |
| 10 – 100 | Strong evidence |
| 3 – 10 | Moderate evidence |
| 1 – 3 | Weak, barely worth mentioning |
| < 1 | Evidence favours Model B |

**CxO translation:** The Bayes Factor is a likelihood ratio between two competing hypotheses. It answers: "How much more likely is the data under Model A than Model B?" BIC selection approximates this without computing it directly — which is why BIC is computationally tractable for large candidate libraries.

In the GR experiment: the winning structure had BIC = −91,009 vs the next best at −90,936. The gap of 73 BIC units corresponds to an astronomically large Bayes Factor in favour of the Einstein structure.

---

## Bootstrap Confidence Interval

Not strictly Bayesian — but used alongside Bayesian selection for uncertainty quantification.

**Method:** Resample the data 500 times with replacement. Refit the model each time. The spread of fitted coefficients gives the uncertainty.

**In the GR result:**
- Fitted k = 18.862 ± 0.025
- 95% CI = [18.812, 18.906]
- True value 6π = 18.850 — inside the CI ✓

**CxO translation:** We are 95% confident the true coefficient lies between 18.81 and 18.91. Since 6π = 18.85 sits comfortably inside that range, the data is consistent with Einstein's exact formula.

---

## Marginal Likelihood (Evidence)

**P(data | Model) = ∫ P(data | θ, Model) · P(θ | Model) dθ**

The integral over all possible parameter values, weighted by the prior.

**CxO translation:** This is the "gold standard" Bayesian model score — but it requires integrating over all parameters, which is computationally expensive. BIC is a large-sample approximation to it. In practice: use BIC for fast model selection, compute marginal likelihood via MCMC only when the decision is high-stakes and the dataset is small.

---

## How These Fit Together in M-W

```
Raw data (Le Verrier's Mercury observations)
        ↓
Prior: 14 dimensionally-valid structural candidates
        ↓
Likelihood: curve-fit each candidate to data
        ↓
BIC selection: rank by posterior approximation
        ↓
Winner: k · GM / (c² · a · (1−e²))
        ↓
Free coefficient fit: k = 18.862
        ↓
Bootstrap CI: [18.812, 18.906]  →  6π = 18.850 ∈ CI  ✓
        ↓
Discovered law: Δφ = 6π·GM / (c²·a·(1−e²))
```

---

## One-Line Glossary

| Term | One line |
|---|---|
| **Prior** | What you knew before the data |
| **Likelihood** | How well the model explains the data |
| **Posterior** | Updated belief after combining prior + data |
| **BIC** | Penalised fit score — lower is better — favours truth |
| **AIC** | Penalised fit score — lower is better — favours prediction |
| **Bayes Factor** | How many times more likely is Model A than Model B |
| **Marginal likelihood** | The gold-standard model score, expensive to compute |
| **Bootstrap CI** | Uncertainty range from resampling the data 500 times |
| **Occam's Razor** | Simpler models win unless complexity earns its keep |

---

*All of the above are standard tools in Bayesian statistics — the M-W framework applies them to physics-informed manifolds rather than traditional regression settings.*

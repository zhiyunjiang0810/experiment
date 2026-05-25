# How Much to Predict: Cost-Aware Predictor Selection for Learning-Augmented Scheduling



## Overview

This repository accompanies *How Much to Predict? Cost-Aware Predictor 
Selection for Learning-Augmented Scheduling*. The paper addresses a 
question absent from prior learning-augmented scheduling (LAS) guarantees: 
**When is a prediction worth its inference cost?**

LAS schedulers improve over non-clairvoyant baselines by using ML 
predictions of job processing times to order jobs. Prior analyses treat 
these predictions as freely available — but in practice every prediction 
requires model inference, and that inference takes time. On a single 
machine, the latency $\tau$ delays *every* one of the $n$ jobs in a 
batch, so accumulated inference cost enters the total completion time 
objective $\sum_j C_j$ as an additive $n\tau$ term. The paper formalizes 
the resulting accuracy–latency tradeoff into a cost-aware predictor-selection 
framework with the following theoretical structure:

- **The $n\tau$ penalty is unavoidable** (Theorems 1–2). Wait-then-schedule 
  achieves $\sum_j C_j \leq n\tau(\theta) + \rho(\eta(\theta))\,\mathrm{OPT}(I)$, 
  and a matching adaptive-adversary lower bound shows that every 
  deterministic non-clairvoyant algorithm pays at least $(n-1)\tau/2$ — 
  even policies that use the inference window $[0, \tau)$ to process 
  jobs cannot escape $\Theta(n\tau)$.

- **The cost-aware optimum scales with workload pressure** (Theorem 3, 
  Corollary 2). Under power-law accuracy–latency tradeoffs, the objective 
  $J(\theta) = n\tau(\theta) + \rho(\eta(\theta))\,\mathrm{OPT}(I)$ is 
  strictly convex; whenever the interior optimum is active, it scales as 
  $\theta^\star \propto (\mathrm{OPT}(I)/n)^{1/(a+b)}$. Heavier workloads 
  justify more complex predictors.
- **A single trace-computable parameter decides when cost matters at all** 
  (Theorem 4). The regime parameter $\kappa = n\tau / \mathrm{OPT}$ 
  measures prediction cost in units of the optimal schedule. A cost-unaware 
  policy that minimizes prediction error alone incurs a permanent 
  $1 + \kappa$ performance ratio that does *not* vanish as $n \to \infty$.

The framework is **algorithm-agnostic**: any LAS scheduler enters only 
through its error–response curve $\rho(\eta)$, independent of internal 
mechanism (SPJF, PRR, permutation-based, etc.). The selection criterion 
therefore applies uniformly across LAS algorithm families.

**This notebook reproduces all main empirical results** on two production 
traces spanning opposite ends of the job-duration spectrum: the **ATLAS** 
Alibaba PAI GPU-cluster trace, where 
$\kappa < 10^{-4}$ even for LLM-class predictors) and the **Azure Functions** 
serverless trace, where a transformer-class 
predictor reaches $\kappa \approx 16\%$). The contrast between the two 
regimes — quantified in Figures 3(a)–(d) and Table 1 — is the central 
empirical finding: cost-aware selection is operationally negligible on 
long-job traces but reshapes the design problem on short-job traces.

## Key Results

The two production traces span opposite ends of the job-duration 
spectrum — a ~5000× gap in effective batch scale — producing 
qualitatively different regimes for cost-aware predictor selection.

| | ATLAS (GPU cluster) | Azure (serverless) |
|:---|:---:|:---:|
| Jobs | 732K | 1.98M |
| Effective scale $p_{\text{eff}}$ | 694 s | 0.13 s |
| Cost-aware optimum $\theta^\star$ (Layer 1) | 34.2 | 22.0 |
| Price of ignoring cost | +20% | +23% |
| Scaling slope (empirical / theory) | 1.24 / 0.61 | 0.61 / 0.57 |
| Per-bucket median $\theta^\star_e / \theta^\star_t$ | 1.41 | **1.02** |
| Regime $\kappa$ at 10 ms/job (transformer-class) | $2.9 \times 10^{-5}$ | **0.16** |

> **The $\kappa$ separation is the central empirical finding.** On ATLAS 
> ($p_{\text{eff}} \approx 700$ s), prediction cost is negligible even for 
> LLM-class predictors ($\kappa < 10^{-4}$). On Azure 
> ($p_{\text{eff}} \approx 0.13$ s), a transformer-class predictor already 
> pushes $\kappa$ to 16% — cost-aware selection becomes the operative 
> design question.

## Repository Structure

```
.
├── experiment_notebook.ipynb   # Main notebook — run on Colab, no GPU needed
├── figure_fixed.py             # 4-panel figure (Figure 1)
├── results_table.py            # Scaling law heatmap table (Table 1)
└── README.md
```

## Data

| Trace | Source | Size | Format |
|:------|:-------|:-----|:-------|
| **ATLAS** (Alibaba PAI) | [GitHub](https://github.com/zhiyunjiang0810/non-clairvoyant-with-predictions) | 732K jobs | 3 × `.tar.gz` → place in `/content/` |
| **Azure Functions 2021** | [Azure Public Dataset](https://github.com/Azure/AzurePublicDataset) | 1.98M invocations | `.rar` → place in `/content/` |

## Reproducing Results

The notebook runs **end-to-end on Google Colab** (free tier, no GPU).

| Cell | What it does | Time |
|:-----|:-------------|-----:|
| 0 | Setup & data extraction | ~10s |
| 1 | Shared helpers | instant |
| 2 | ATLAS data loading | ~30s |
| 3 | ATLAS Layer 1 (U-shape + scaling law) | ~2 min |
| 4 | ATLAS Layer 2 (GBT family, 12 configs) | ~15 min |
| 5 | Azure Layer 1 (U-shape + scaling law) | ~2 min |
| 6 | Plot Figure 1 | instant |

**Total: ~20 minutes.**

```bash
# Or run locally:
pip install numpy pandas scipy lightgbm matplotlib rarfile
jupyter notebook experiment_notebook.ipynb
```

## Dependencies

```
numpy, pandas, scipy, lightgbm, matplotlib, rarfile
```

All pip-installable. The notebook installs missing packages automatically.

## License

Released for anonymous review. Full license will be added upon acceptance.

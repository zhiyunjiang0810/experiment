# The Price of Prediction: Cost-Aware Model Selection for Learning-Augmented Scheduling



## Overview

Learning-augmented schedulers use ML predictions to order jobs, but **inference isn't free**. A more accurate predictor takes longer to run, and that latency delays *all* n jobs — adding an nτ(θ) penalty to total completion time.

This paper asks: **given the accuracy–latency tradeoff, which predictor should be used?**

We provide:
- An **algorithm-agnostic framework** that depends on the scheduler only through its error-response curve ρ(η)
- A proof that inference latency contributes an **unavoidable Θ(nτ) penalty** (adaptive-adversary lower bound)  
- A **power-law scaling law** for the optimal predictor complexity θ*
- A **regime parameter κ = nτ/OPT** that tells practitioners when cost-aware selection matters — computable from the workload alone
- Validation on **two production traces** spanning a 5,000× range in job duration

## Key Results

| | ATLAS (GPU jobs) | Azure (serverless) |
|:---|:---:|:---:|
| Jobs | 732K | 1.98M |
| p_eff | 694 s | 0.13 s |
| θ\* (Layer 1) | 34.2 | 22.0 |
| Price of ignoring cost | 21% | 23% |
| Scaling slope (emp / thy) | 1.24 / 0.61 | 0.61 / 0.57 |
| Per-bucket median ratio | 1.41 | **1.02** |
| κ (Transformer, 10ms) | 3 × 10⁻⁵ | **0.16** |

> On ATLAS, prediction cost is negligible at all scales (κ < 10⁻⁴).  
> On Azure, a transformer-class predictor already reaches κ ≈ 16% — cost-aware selection is essential.

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

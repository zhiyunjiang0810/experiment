# The Price of Prediction: Cost-Aware Model Selection for Learning-Augmented Scheduling

Code for reproducing experiments in the NeurIPS 2026 submission.

## Overview

This paper studies how prediction inference cost affects predictor selection in learning-augmented scheduling. When a scheduler uses ML predictions to order jobs, the inference latency τ(θ) of the predictor delays *all* n jobs, adding an nτ(θ) penalty to the total completion time. We provide a framework for selecting predictor complexity θ that balances prediction quality against this cost, and validate the theory on two production traces.

## Repository structure

```
├── experiment_notebook.ipynb   # Main notebook (run on Colab)
├── figure_fixed.py             # 4-panel figure (Fig. 1)
├── results_table.py            # Scaling law table (Table 1)
└── README.md
```

## Data

**ATLAS (Alibaba PAI GPU trace)**
- Source: [Alibaba PAI Trace](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2023)
- 732,355 completed GPU-training jobs
- Download the three `.tar.gz` files and place in `/content/`

**Azure Functions 2021**
- Source: [Azure Public Dataset](https://github.com/Azure/AzurePublicDataset)
- 1.98M serverless invocations over two weeks
- Download the `.rar` archive and place in `/content/`

## Reproducing results

The notebook runs end-to-end on Google Colab (free tier, no GPU needed).

```
Cell 0   Setup & data extraction          ~10s
Cell 1   Shared helpers                   instant
Cell 2   ATLAS data loading               ~30s
Cell 3   ATLAS Layer 1 (U-shape + scaling law)   ~2 min
Cell 4   ATLAS Layer 2 (GBT family)       ~15 min
Cell 5   Azure Layer 1 (U-shape + scaling law)   ~2 min
Cell 6   Plot Figure 1                    instant
```

Total wall time: ~20 minutes.

## Dependencies

```
numpy
pandas
scipy
lightgbm
matplotlib
rarfile
```


## License

Released for review purposes. Full license will be added upon acceptance.

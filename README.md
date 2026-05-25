# How Much to Predict: Cost-Aware Predictor Selection for Learning-Augmented Scheduling

## Overview

This repository contains the experimental notebook for the paper *How Much to Predict? Cost-Aware Predictor Selection for Learning-Augmented Scheduling*.

The paper studies a basic but previously overlooked question in learning-augmented scheduling: **when is a prediction worth its inference cost?** In single-machine scheduling with objective \(\sum_j C_j\), prediction latency is not free. If a predictor takes \(\tau\) time per batch, that delay affects all \(n\) jobs, contributing an additive \(n\tau\) term to the objective. The paper develops a cost-aware predictor-selection framework that balances prediction quality against inference latency.

At a high level, the repository reproduces the paper’s main empirical claims:

- inference cost induces an unavoidable \(\Theta(n\tau)\) penalty,
- the cost-aware optimal predictor complexity depends on workload pressure,
- the regime parameter \(\kappa = n\tau / \mathrm{OPT}\) determines when prediction cost is negligible and when it materially changes the design choice.

The notebook compares two production traces with very different time scales:

- **ATLAS / Alibaba PAI GPU trace**: long jobs, where prediction cost is typically negligible,
- **Azure Functions 2021 trace**: short jobs, where prediction cost can be substantial.

## What is in this repository

This repo is centered around a single notebook that runs the experiments end to end:

- `experiment_notebook.ipynb` — main notebook for data loading, synthetic predictor experiments, LightGBM experiments, and figure generation.

The notebook currently includes six major stages:

1. **Setup and Azure extraction**
   - installs `rarfile` and `lightgbm`,
   - extracts the Azure `.rar` archive into `azure_data/`.

2. **Shared helper functions**
   - computes `OPT` for single-machine \(\sum C_j\),
   - evaluates SPJF under predicted processing times,
   - fits power laws,
   - defines the semi-synthetic predictor used in Layer 1.

3. **ATLAS Layer 1**
   - loads and joins the three Alibaba PAI tables,
   - constructs job-level durations,
   - runs the semi-synthetic U-shape and workload-pressure experiments.

4. **ATLAS Layer 2**
   - trains a family of LightGBM predictors,
   - measures empirical prediction latency,
   - evaluates the cost-aware objective across predictor configurations.

5. **Azure Layer 1**
   - loads the Azure Functions trace,
   - runs the same semi-synthetic cost-aware experiments,
   - computes the \(\kappa\)-regime comparison table.

6. **Figure generation**
   - produces the four-panel figure and saves:
     - `figure_4panel.pdf`
     - `figure_4panel.png`

## Repository structure

```text
.
├── README.md
└── experiment_notebook.ipynb
```

## Data requirements

The notebook expects the raw trace files to be available locally, with paths matching the code.

### 1. ATLAS / Alibaba PAI trace

Place these three archives in `/content/` when running on Google Colab:

- `/content/pai_group_tag_table.tar.gz`
- `/content/pai_job_table.tar.gz`
- `/content/pai_task_table.tar.gz`

The notebook extracts them into:

- `/content/extracted/`

and expects the extracted CSV files:

- `pai_group_tag_table.csv`
- `pai_job_table.csv`
- `pai_task_table.csv`

### 2. Azure Functions 2021 trace

Place the Azure archive at:

- `/content/AzureFunctionsInvocationTraceForTwoWeeksJan2021 (1).rar`

The notebook extracts it into:

- `azure_data/`

and then reads:

- `azure_data/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt`

## How to run

### Google Colab

The notebook is written primarily for Colab-style paths such as `/content/...`.

1. Upload the three ATLAS `.tar.gz` files to `/content/`.
2. Upload the Azure `.rar` file to `/content/`.
3. Open `experiment_notebook.ipynb`.
4. Run the cells in order.

### Local execution

To run locally, you will likely need to adjust the hard-coded `/content/...` paths in the notebook.

Install dependencies:

```bash
pip install numpy pandas scipy lightgbm matplotlib rarfile
```

You also need an `unrar` binary available on your system if you want to extract the Azure archive in the same way as the notebook.

Then launch Jupyter:

```bash
jupyter notebook experiment_notebook.ipynb
```

## Outputs

Running the notebook produces:

- printed summaries for ATLAS Layer 1,
- printed summaries for ATLAS Layer 2 with LightGBM latency and cost-aware comparisons,
- printed summaries for Azure Layer 1,
- a saved four-panel figure:
  - `figure_4panel.pdf`
  - `figure_4panel.png`

## Experimental scope

Based on the current code, this repository reproduces:

- the semi-synthetic cost-aware experiments for **ATLAS** and **Azure**,
- the workload-pressure scaling analysis,
- the empirical \(\kappa\)-regime comparison,
- a real-predictor **LightGBM** study for **ATLAS**,
- the final four-panel figure generated from these results.

The repository does **not** currently contain separate standalone scripts such as `figure_fixed.py` or `results_table.py`; the analysis is implemented directly inside the notebook.

## Dependencies

The notebook imports or installs the following Python packages:

- `numpy`
- `pandas`
- `scipy`
- `lightgbm`
- `matplotlib`
- `rarfile`

It also uses standard-library modules including:

- `os`
- `time`
- `pathlib`
- `tarfile`
- `gc`
- `warnings`
- `dataclasses`
- `typing`

## Notes

- No GPU is required for the notebook as written.
- The Azure extraction cell uses `apt-get install -y unrar` and is therefore easiest to run in Colab or another Debian-like environment.
- The current notebook filename in the material you shared appears as `experiment_notebook (2).ipynb`; if you keep that name in the repository, update the README references accordingly.

## Citation

If you use this repository, please cite the corresponding paper:

*How Much to Predict? Cost-Aware Predictor Selection for Learning-Augmented Scheduling*.

## License

Released for anonymous review. A formal license can be added later if needed.

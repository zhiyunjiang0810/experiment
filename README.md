# How Much to Predict? Cost-Aware Predictor Selection for Learning-Augmented Scheduling

## Overview

This repository accompanies *How Much to Predict? Cost-Aware Predictor 
Selection for Learning-Augmented Scheduling*. The paper addresses a 
question absent from prior learning-augmented scheduling (LAS) guarantees: 
**when does inference latency change the scheduling decision?**

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
Alibaba PAI GPU-cluster trace, where $\kappa < 10^{-4}$ even for 
LLM-class predictors, and the **Azure Functions** serverless trace, where 
a transformer-class predictor reaches $\kappa \approx 16\%$. The contrast 
between the two regimes — quantified in Figures 3(a)–(d) and Table 1 — 
is the central empirical finding: cost-aware selection is operationally 
negligible on long-job traces but reshapes the design problem on 
short-job traces.

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

```text
.
├── README.md
└── experiment_notebook.ipynb
```

## Data

| Trace | Source | Files |
|:------|:-------|:------|
| **ATLAS** (Alibaba PAI) | [GitHub](https://github.com/zhiyunjiang0810/non-clairvoyant-with-predictions) | 3 × `.tar.gz` (732K jobs) |
| **Azure Functions 2021** | [Azure Public Dataset](https://github.com/Azure/AzurePublicDataset) | 1 × `.rar` (1.98M invocations) |

**Placement** (Google Colab paths; adjust for local runs):

```text
/content/
├── pai_group_tag_table.tar.gz    ← ATLAS
├── pai_job_table.tar.gz
├── pai_task_table.tar.gz
└── AzureFunctionsInvocationTraceForTwoWeeksJan2021 (1).rar  ← Azure
```

The notebook extracts these automatically on first run.

## Data Requirements

The notebook expects raw trace files to be available locally at the paths used in the code.

### 1. ATLAS / Alibaba PAI trace

Place these three archives in `/content/` when running on Google Colab:

- `/content/pai_group_tag_table.tar.gz`
- `/content/pai_job_table.tar.gz`
- `/content/pai_task_table.tar.gz`

The notebook extracts them into `/content/extracted/` and expects:

- `pai_group_tag_table.csv`
- `pai_job_table.csv`
- `pai_task_table.csv`

### 2. Azure Functions 2021 trace

Place the Azure archive at:

- `/content/AzureFunctionsInvocationTraceForTwoWeeksJan2021 (1).rar`

The notebook extracts it into `azure_data/` and reads:

- `azure_data/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt`

## How to Run

The repository is centered around `experiment_notebook.ipynb`, which runs the experiments end to end.

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

- printed summaries for the ATLAS experiments,
- printed summaries for the Azure experiments,
- Figure 3 panels (a)–(d), saved as:
  - `figure_4panel.pdf`
  - `figure_4panel.png`
- Table 1 scaling-law validation outputs.

## Dependencies

- Python >= 3.8
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

## Citation

If you use this repository, please cite the corresponding paper:

*How Much to Predict? Cost-Aware Predictor Selection for Learning-Augmented Scheduling*.

## License

Released for anonymous review. A formal license can be added later if needed.

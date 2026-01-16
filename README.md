# Praxis_BGM_R_interface

An R wrapper around the Python Praxis-BGM package using **reticulate**. This repo provides
an R package-style structure with an R function (`R/praxis_bgm_fit.R`) that loads a conda
environment and exposes a convenient `praxis_bgm_fit()` helper, plus an R Markdown
tutorial (`Praxis_R_Turtorial.Rmd`).

## About Praxis-BGM (Python/JAX)

The original Praxis-BGM Python implementation is available at:
https://github.com/ContiLab-usc/Praxis-BGM

Praxis-BGM (Prior-Augmented Bayesian Gaussian Mixture Model via Natural-Gradient Variational
Inference) is a **semi-supervised transfer-learning framework** for clustering high-dimensional
omics, multi-omics, and single-cell data using **Bayesian Gaussian Mixture Models** with
**Natural-Gradient Variational Inference (NGVI)**. It enables incorporation of
**cluster-specific priors** (means, covariances, sparsity masks, and mixing weights) from a
labeled source dataset to guide clustering in an unlabeled target dataset. The method is
implemented in **JAX** for GPU/TPU acceleration and numerically stable updates.

### Features (from the Python implementation)

- Semi-supervised Bayesian transfer learning for Gaussian mixture models
- NGVI (VON algorithm) for fast, stable optimization
- Priors on means `μ`, covariances `Σ` (optional), structural adjacency masks `A` (optional),
  and mixing weights `θ` (optional)
- Structural sparsity masks `A` to encode pathway or network knowledge
- Mini-batch training for large n
- Bayes factor–based feature importance scoring
- Compatible with high-dimensional omics (d > 1,000)

## Prerequisites

- R (with the **reticulate** package installed).
- A working Python environment (conda) that can run Praxis-BGM and JAX.

## 1) Install Praxis-BGM and dependencies (Python)

> **Important:** Install the Python package **before** calling the R wrapper.

Create a conda environment (example name: `jax_env`) and install requirements including the Python version of Praxis-BGM as a dependency. Adjust to
match your Praxis-BGM installation instructions.

```bash
conda create -n jax_env python=3.10 -y
conda activate jax_env
pip install jax jaxlib numpy
pip install git+https://github.com/ContiLab-usc/Praxis-BGM.git
```

### Python requirements 

- python >= 3.9
- jax >= 0.4.20
- jaxlib >= 0.4.20
- numpy
- scikit-learn
- matplotlib

## 2) Use the R wrapper

Source the wrapper script in R. The script uses:

```r
library(reticulate)
use_condaenv("jax_env")
```

Then call the wrapper function:

```r
result <- praxis_bgm_fit(
  data = your_matrix,
  K = 3,
  seed = 123
)
```

Returned elements include posterior parameters, responsibilities, and the model object.

## Tutorial

`Praxis_R_Turtorial.Rmd` is an end-to-end tutorial that sources the R script, simulates
Gaussian-mixture data, fits Praxis-BGM, and inspects outputs.

The original Python tutorial notebook is here:
https://github.com/ContiLab-usc/Praxis-BGM/blob/main/Praxis_BGM_Tutorial.ipynb

## Files

- `R/praxis_bgm_fit.R`: R script that defines `praxis_bgm_fit()` for use with reticulate.
- `R/praxis_simulation.R`: R helpers for simulating source/target data in the tutorial/tests.
- `Praxis_R_Turtorial.Rmd`: Tutorial that sources the R script and demonstrates fitting.
- `DESCRIPTION` / `NAMESPACE`: Package metadata for the R wrapper.

## Notes

- Ensure the conda environment name matches the one in the template.
- If you use a different environment manager, replace `use_condaenv()` with the matching
  reticulate configuration (e.g., `use_python()` or `use_virtualenv()`).

## Citation

If you use Praxis-BGM in your research, please cite:

```
@article{jia2025praxisbgm,
  title={Clustering of Omic Data Using Semi-Supervised Transfer Learning for Gaussian Mixture Models via Natural-Gradient Variational Inference},
  author={Jia, Qiran and Goodrich, Jesse A. and Conti, David V.},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.11.13.688299},
}
```

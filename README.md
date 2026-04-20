# Praxis-BGM R Interface

This repository provides an R interface to the Python/JAX implementation of
**Praxis-BGM** through [`reticulate`](https://rstudio.github.io/reticulate/).
The interface has been updated to match the current Python package API and is
now maintained from a single script:

- `R/praxis_bgm_interface.R`

That script contains:

- `praxis_bgm_fit()` for fitting the updated `praxis_bgm.Praxis_BGM` model
- `praxis_bgm_bf_selection()` for Bayes-factor feature selection
- simulation helpers used by the tutorial and tests

## Python package status reflected here

The R wrapper now matches the updated Python constructor and runtime arguments,
including:

- `prior_weights` instead of the older `prior_pis`
- `init_pis` for initialization weights
- `freeze_A_zeros` instead of the older mask-enforcement arguments
- runtime controls such as `data_precision_int`, `likelihood_temp`, `rho_prec`,
  `rho_mu`, and `elbo_eval_freq`

The Bayes-factor selection logic is unchanged from the Python implementation.
The R wrapper simply exposes it more cleanly and returns both:

- R-friendly one-based feature indices
- raw zero-based Python indices for exact parity with the source implementation

## About Praxis-BGM

The original Python implementation is available at:

- [ContiLab-usc/Praxis-BGM](https://github.com/ContiLab-usc/Praxis-BGM)

Praxis-BGM is a prior-augmented Bayesian Gaussian mixture model for
semi-supervised transfer learning. It allows source-domain information such as
cluster means, covariances, sparsity structure, and mixture weights to guide
clustering in an unlabeled target dataset.

## Requirements

### Python

Install the Praxis-BGM Python package into an environment that `reticulate` can
see. The examples in this repository assume a conda environment named
`Praxis_env`.

```bash
conda create -n Praxis_env python=3.10 -y
conda activate Praxis_env
pip install jax jaxlib numpy scikit-learn matplotlib
pip install git+https://github.com/ContiLab-usc/Praxis-BGM.git
```

### R

Required R packages:

- `reticulate`
- `MASS`
- `proxy`
- `clue`

Suggested for the tutorial:

- `mclust`
- `rmarkdown`

## Quick start

Point `reticulate` at the Python environment, then source the single wrapper
script:

```r
library(reticulate)

praxis_python <- Sys.getenv("RETICULATE_PYTHON", unset = "")
if (nzchar(praxis_python)) {
  use_python(praxis_python, required = TRUE)
} else {
  use_condaenv("Praxis_env", required = TRUE)
}

source("R/praxis_bgm_interface.R")
```

If you created the environment by full path, you can point `reticulate` at it
directly before starting R:

```bash
export RETICULATE_PYTHON=/Users/qiranjia19961112/.conda/envs/Praxis_env/bin/python
```

Fit a model:

```r
fit <- praxis_bgm_fit(
  data = your_matrix,
  K = 3,
  seed = 123,
  prior_weights = c(1 / 3, 1 / 3, 1 / 3),
  num_iters = 50,
  batch_size = min(50, nrow(your_matrix)),
  verbose = FALSE
)
```

Important returned components include:

- `assignments`: 1-based cluster labels for R users
- `assignments_zero_based`: raw Python cluster labels
- `learned_weights`: fitted mixture weights
- `posterior_mus`, `posterior_covs`, `posterior_pis`, `responsibilities`
- `model_summary`, `elbo_history`, and the live Python `model`

Run Bayes-factor feature selection with the dedicated wrapper:

```r
bf <- praxis_bgm_bf_selection(
  model = fit,
  data = your_matrix,
  top_n = 20,
  visual = FALSE
)
```

The feature-selection result includes:

- `top_features`: one-based feature indices
- `top_features_zero_based`: raw Python indices
- `classification`: one-based Jeffreys-scale buckets
- `classification_zero_based`: raw Python Jeffreys-scale buckets

## Tutorials

The repository includes two updated R Markdown documents:

- `Praxis_R_Wrapper.Rmd`: documents the updated wrapper design and a minimal example
- `Praxis_R_Tutorial.Rmd`: end-to-end tutorial with simulation, transferred priors,
  and `praxis_bgm_bf_selection()`

The tutorial follows the updated wrapper rather than the older `R/` file layout.

## Repository layout

- `R/praxis_bgm_interface.R`: single source-of-truth R interface
- `Praxis_R_Wrapper.Rmd`: wrapper notes and compact example
- `Praxis_R_Tutorial.Rmd`: full tutorial
- `tests/testthat/test-praxis_bgm_fit.R`: basic tests for the wrapper and helpers

## Citation

If you use Praxis-BGM in your research, please cite:

```bibtex
@article{jia2025praxisbgm,
  title={Clustering of Omic Data Using Semi-Supervised Transfer Learning for Gaussian Mixture Models via Natural-Gradient Variational Inference},
  author={Jia, Qiran and Goodrich, Jesse A. and Conti, David V.},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.11.13.688299},
}
```

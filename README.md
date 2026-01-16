# Praxis_BGM_R_interface

An R wrapper around the Python Praxis-BGM package using **reticulate**. This repo provides
an R script (`Praxis_R_Wrapper.R`) that loads a conda environment and exposes a convenient
`praxis_bgm_fit()` helper, plus an R Markdown tutorial (`Praxis_R_Turtorial.Rmd`).

## Prerequisites

- R (with the **reticulate** package installed).
- A working Python environment that can run Praxis-BGM and JAX.

## 1) Install Praxis-BGM and dependencies (Python)

> **Important:** Install the Python package **before** calling the R wrapper.

Create a conda environment (example name: `jax_env`) and install requirements. Adjust to
match your Praxis-BGM installation instructions.

```bash
conda create -n jax_env python=3.10 -y
conda activate jax_env
pip install praxis_bgm jax jaxlib numpy
```

If Praxis-BGM is installed from source, follow the upstream installation guide and ensure
it is available in the same conda environment used by reticulate.

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

## Files

- `Praxis_R_Wrapper.R`: R script that defines `praxis_bgm_fit()` for use with reticulate.
- `Praxis_R_Turtorial.Rmd`: Tutorial that sources the R script and demonstrates fitting.

## Notes

- Ensure the conda environment name matches the one in the template.
- If you use a different environment manager, replace `use_condaenv()` with the matching
  reticulate configuration (e.g., `use_python()` or `use_virtualenv()`).

library(testthat)
library(reticulate)

source("R/praxis_bgm_fit.R")
source("R/praxis_simulation.R")

skip_if_no_modules <- function() {
  if (!py_module_available("praxis_bgm")) {
    skip("praxis_bgm Python module not available")
  }
  if (!py_module_available("jax")) {
    skip("jax Python module not available")
  }
}

test_that("praxis_bgm_fit validates required inputs", {
  skip_if_no_modules()

  expect_error(praxis_bgm_fit(), "`data` is required")
  expect_error(praxis_bgm_fit(data = matrix(1, nrow = 2), K = 0), "positive integer")
})

test_that("simulation helpers return expected shapes", {
  skip_if_not_installed("MASS")
  skip_if_not_installed("proxy")
  skip_if_not_installed("clue")

  sim <- generate_overlapping_gmm_samples(
    n_components = 3,
    n_causal = 4,
    n_features = 10,
    n_samples = 60,
    random_seed = 42
  )

  expect_equal(dim(sim$samples), c(60, 10))
  expect_equal(length(sim$labels), 60)
  expect_equal(dim(sim$full_means), c(3, 10))

  shifted <- randomly_shift_means(sim$full_means, shift_magnitude = 0.5, percentage = 0.2, random_seed = 1)
  expect_equal(dim(shifted), dim(sim$full_means))
  expect_false(all(shifted == sim$full_means))

  expect_equal(l2_norm_with_alignment(sim$full_means, sim$full_means), 0)
})

test_that("praxis_bgm_fit runs with different priors and sizes", {
  skip_if_not_installed("MASS")
  skip_if_no_modules()

  sim_small <- generate_overlapping_gmm_samples(
    n_components = 2,
    n_causal = 3,
    n_features = 6,
    n_samples = 20,
    random_seed = 7
  )

  result <- praxis_bgm_fit(
    data = sim_small$samples,
    K = 2,
    seed = 7,
    prior_mus = sim_small$full_means,
    num_iters = 5,
    batch_size = 10,
    verbose = FALSE
  )

  expect_true(is.list(result))
  expect_true(all(c("posterior_mus", "posterior_covs", "posterior_pis", "responsibilities", "model") %in% names(result)))
})

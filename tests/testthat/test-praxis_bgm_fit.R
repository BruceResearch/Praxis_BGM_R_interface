library(testthat)
library(reticulate)

source(testthat::test_path("..", "..", "R", "praxis_bgm_interface.R"))

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

  expect_error(praxis_bgm_fit(K = 2), "`data` is required")
  expect_error(praxis_bgm_fit(data = matrix(1, nrow = 2), K = 0), "greater than or equal to 2")
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
  expect_true(all(
    c(
      "assignments",
      "assignments_zero_based",
      "learned_weights",
      "posterior_mus",
      "posterior_covs",
      "posterior_pis",
      "responsibilities",
      "model_summary",
      "elbo_history",
      "model"
    ) %in% names(result)
  ))
})

test_that("praxis_bgm_bf_selection validates inputs", {
  skip_if_no_modules()

  expect_error(
    praxis_bgm_bf_selection(model = list(), data = matrix(1, nrow = 2), top_n = 0),
    "`top_n` must be a positive integer"
  )
})

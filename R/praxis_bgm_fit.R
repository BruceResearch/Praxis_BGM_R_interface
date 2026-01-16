library(reticulate)

use_condaenv("jax_env")

#' Fit Praxis-BGM from R via reticulate
#'
#' @param data Numeric matrix or data frame with observations in rows.
#' @param K Integer number of clusters.
#' @param seed Integer seed for JAX PRNG.
#' @param prior_mus Optional prior means (K x P).
#' @param prior_Sigmas Optional prior covariances (K x P x P).
#' @param init_mus Optional initialization means (K x P).
#' @param init_covs Optional initialization covariances (K x P x P).
#' @param prior_pis Optional prior mixing weights.
#' @param beta Numeric precision regularization.
#' @param tol Numeric convergence tolerance.
#' @param max_iters Integer maximum iterations for internal initialization.
#' @param verbose Logical; enable verbose logging.
#' @param sparse_A Optional sparsity mask matrix.
#' @param cluster_A Optional cluster-level mask matrix.
#' @param prior_mus_variance Numeric variance for mean priors.
#' @param num_samples Integer number of Monte Carlo samples.
#' @param enforce_mask Logical; whether to enforce mask constraints.
#' @param mask_space Character; mask space type.
#' @param spd_eps Numeric jitter for SPD stabilization.
#' @param num_iters Integer number of fit iterations.
#' @param batch_size Integer batch size.
#' @param early_stop Logical; enable early stopping.
#' @param patience Integer early stop patience.
#'
#' @return List containing posterior parameters and the fitted model.
#' @export
praxis_bgm_fit <- function(
  data,
  K,
  seed = 0L,
  prior_mus = NULL,
  prior_Sigmas = NULL,
  init_mus = NULL,
  init_covs = NULL,
  prior_pis = NULL,
  beta = 0.001,
  tol = 1e-4,
  max_iters = 1000,
  verbose = TRUE,
  sparse_A = NULL,
  cluster_A = NULL,
  prior_mus_variance = 10.0,
  num_samples = 100,
  enforce_mask = FALSE,
  mask_space = "precision",
  spd_eps = 1e-6,
  num_iters = 100,
  batch_size = 50,
  early_stop = FALSE,
  patience = 2
) {
  if (!reticulate::py_module_available("praxis_bgm")) {
    stop("The 'praxis_bgm' Python module is not available. Install it before use.")
  }
  if (!reticulate::py_module_available("jax")) {
    stop("The 'jax' Python module is not available. Install it before use.")
  }
  if (missing(data)) {
    stop("`data` is required and must be a numeric matrix or data frame.")
  }
  if (!is.numeric(K) || length(K) != 1 || K <= 0) {
    stop("`K` must be a positive integer.")
  }

  as_numpy <- function(x, np, dtype = "float32") {
    if (is.null(x)) {
      return(NULL)
    }
    if (inherits(x, "python.builtin.object")) {
      return(x)
    }
    np$array(x, dtype = dtype)
  }

  jax <- reticulate::import("jax", delay_load = TRUE)
  np <- reticulate::import("numpy", delay_load = TRUE)
  praxis <- reticulate::import("praxis_bgm", delay_load = TRUE)

  data_matrix <- if (is.data.frame(data)) {
    as.matrix(data)
  } else {
    data
  }
  if (!is.matrix(data_matrix) && length(dim(data_matrix)) != 2) {
    stop("`data` must be a 2D matrix or data frame.")
  }
  if (!is.numeric(data_matrix)) {
    stop("`data` must contain numeric values.")
  }

  rng_key <- jax$random$PRNGKey(as.integer(seed))

  model <- praxis$Praxis_BGM(
    rng_key = rng_key,
    K = as.integer(K),
    prior_mus = as_numpy(prior_mus, np),
    prior_Sigmas = as_numpy(prior_Sigmas, np),
    init_mus = as_numpy(init_mus, np),
    init_covs = as_numpy(init_covs, np),
    prior_pis = as_numpy(prior_pis, np),
    beta = beta,
    tol = tol,
    max_iters = as.integer(max_iters),
    verbose = verbose,
    sparse_A = as_numpy(sparse_A, np),
    cluster_A = as_numpy(cluster_A, np),
    prior_mus_variance = prior_mus_variance,
    num_samples = as.integer(num_samples),
    enforce_mask = enforce_mask,
    mask_space = mask_space,
    spd_eps = spd_eps
  )

  model$fit(
    as_numpy(data_matrix, np),
    num_iters = as.integer(num_iters),
    batch_size = as.integer(batch_size),
    early_stop = early_stop,
    patience = as.integer(patience)
  )

  posteriors <- model$get_posteriors(as_numpy(data_matrix, np))

  list(
    posterior_mus = posteriors[[1]],
    posterior_covs = posteriors[[2]],
    posterior_pis = posteriors[[3]],
    responsibilities = posteriors[[4]],
    model = model
  )
}

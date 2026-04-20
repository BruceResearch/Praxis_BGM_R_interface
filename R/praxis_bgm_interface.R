#' Validate that the required Python modules are available
#'
#' @keywords internal
.praxis_bgm_assert_python_modules <- function() {
  if (!reticulate::py_module_available("praxis_bgm")) {
    stop("The Python module `praxis_bgm` is not available in the active reticulate environment.")
  }
  if (!reticulate::py_module_available("jax")) {
    stop("The Python module `jax` is not available in the active reticulate environment.")
  }
}

#' Convert R objects to NumPy arrays when needed
#'
#' @keywords internal
.praxis_bgm_as_numpy <- function(x, np, dtype = "float32") {
  if (is.null(x)) {
    return(NULL)
  }
  if (inherits(x, "python.builtin.object")) {
    return(x)
  }
  np$array(x, dtype = dtype)
}

#' Convert Python arrays back to R objects
#'
#' @keywords internal
.praxis_bgm_py_array_to_r <- function(x, np) {
  if (is.null(x)) {
    return(NULL)
  }
  if (inherits(x, "python.builtin.object")) {
    return(reticulate::py_to_r(np$asarray(x)))
  }
  x
}

#' Extract an element from a Python tuple/list or an R list
#'
#' @keywords internal
.praxis_bgm_extract_sequence <- function(x, index_zero_based) {
  if (inherits(x, "python.builtin.object")) {
    return(x[[as.integer(index_zero_based)]])
  }
  x[[as.integer(index_zero_based) + 1L]]
}

#' Convert input data into a validated numeric matrix
#'
#' @keywords internal
.praxis_bgm_validate_matrix <- function(data, arg = "data") {
  if (missing(data)) {
    stop(sprintf("`%s` is required and must be a numeric matrix or data frame.", arg))
  }

  data_matrix <- if (is.data.frame(data)) {
    data.matrix(data)
  } else {
    as.matrix(data)
  }

  if (!is.matrix(data_matrix) || length(dim(data_matrix)) != 2L) {
    stop(sprintf("`%s` must be a two-dimensional matrix or data frame.", arg))
  }
  if (!is.numeric(data_matrix)) {
    stop(sprintf("`%s` must contain numeric values.", arg))
  }
  if (!all(is.finite(data_matrix))) {
    stop(sprintf("`%s` must contain only finite values.", arg))
  }

  storage.mode(data_matrix) <- "double"
  data_matrix
}

#' Resolve a fitted Praxis-BGM Python model
#'
#' @keywords internal
.praxis_bgm_resolve_model <- function(model) {
  model_obj <- if (is.list(model) && !is.null(model$model)) {
    model$model
  } else {
    model
  }

  if (!inherits(model_obj, "python.builtin.object")) {
    stop("`model` must be a Python Praxis-BGM object or a list returned by `praxis_bgm_fit()`.")
  }

  model_obj
}

#' Shift zero-based feature indices to one-based R indices
#'
#' @keywords internal
.praxis_bgm_shift_index_list <- function(x) {
  stats::setNames(
    lapply(x, function(indices) as.integer(unlist(indices)) + 1L),
    names(x)
  )
}

#' Fit Praxis-BGM from R via reticulate
#'
#' This wrapper matches the updated Python `praxis_bgm.Praxis_BGM` API and
#' returns R-friendly outputs while keeping the fitted Python model available
#' for advanced downstream use.
#'
#' @param data Numeric matrix or data frame with observations in rows.
#' @param K Integer number of clusters.
#' @param seed Integer seed used to create the JAX PRNG key.
#' @param prior_mus Optional prior means with shape `K x P`.
#' @param prior_Sigmas Optional prior covariance array with shape `K x P x P`.
#' @param prior_weights Optional prior mixture weights of length `K`.
#' @param init_mus Optional initialization means with shape `K x P`.
#' @param init_covs Optional initialization covariances with shape `K x P x P`.
#' @param init_pis Optional initialization weights of length `K`.
#' @param beta Positive numeric regularization constant.
#' @param tol Non-negative convergence tolerance.
#' @param max_iters Integer maximum number of initialization iterations.
#' @param verbose Logical; whether to print the Python model's progress logs.
#' @param sparse_A Optional sparsity mask matrix.
#' @param cluster_A Optional cluster-specific adjacency mask.
#' @param freeze_A_zeros Logical; freeze zero-valued `A` entries during fitting.
#' @param prior_mus_variance Numeric variance used when means are initialized automatically.
#' @param num_samples Integer number of Monte Carlo samples.
#' @param data_precision_int Optional positive integer used by the Python implementation.
#' @param likelihood_temp Positive numeric likelihood temperature.
#' @param rho_prec Positive numeric damping parameter for precision updates.
#' @param rho_mu Positive numeric damping parameter for mean updates.
#' @param elbo_eval_freq Positive integer ELBO evaluation frequency.
#' @param num_iters Positive integer number of optimization iterations.
#' @param batch_size Optional positive integer mini-batch size. Defaults to `min(50, nrow(data))`.
#' @param early_stop Logical; whether to enable early stopping.
#' @param patience Positive integer patience used when `early_stop = TRUE`.
#'
#' @return A list containing 1-based and zero-based cluster assignments, learned
#'   weights, posterior parameters, responsibilities, model metadata, ELBO
#'   history, and the live Python model object.
#' @export
praxis_bgm_fit <- function(
  data,
  K,
  seed = 0L,
  prior_mus = NULL,
  prior_Sigmas = NULL,
  prior_weights = NULL,
  init_mus = NULL,
  init_covs = NULL,
  init_pis = NULL,
  beta = 1e-3,
  tol = 1e-4,
  max_iters = 1000L,
  verbose = TRUE,
  sparse_A = NULL,
  cluster_A = NULL,
  freeze_A_zeros = FALSE,
  prior_mus_variance = 1.0,
  num_samples = 100L,
  data_precision_int = NULL,
  likelihood_temp = 1.0,
  rho_prec = 0.05,
  rho_mu = 1.0,
  elbo_eval_freq = 10L,
  num_iters = 50L,
  batch_size = NULL,
  early_stop = FALSE,
  patience = 2L
) {
  .praxis_bgm_assert_python_modules()

  if (!is.numeric(K) || length(K) != 1L || is.na(K) || K < 2 || K != as.integer(K)) {
    stop("`K` must be a single integer greater than or equal to 2.")
  }

  data_matrix <- .praxis_bgm_validate_matrix(data)
  n_obs <- nrow(data_matrix)

  if (is.null(batch_size)) {
    batch_size <- min(50L, n_obs)
  }
  if (!is.numeric(batch_size) || length(batch_size) != 1L || is.na(batch_size) ||
      batch_size < 1 || batch_size != as.integer(batch_size)) {
    stop("`batch_size` must be a positive integer.")
  }
  if (batch_size > n_obs) {
    stop("`batch_size` cannot exceed the number of rows in `data`.")
  }

  np <- reticulate::import("numpy", delay_load = TRUE, convert = FALSE)
  jax <- reticulate::import("jax", delay_load = TRUE, convert = FALSE)
  praxis <- reticulate::import("praxis_bgm", delay_load = TRUE, convert = FALSE)

  rng_key <- jax$random$PRNGKey(as.integer(seed))
  data_numpy <- .praxis_bgm_as_numpy(data_matrix, np)

  model <- praxis$Praxis_BGM(
    rng_key = rng_key,
    K = as.integer(K),
    prior_mus = .praxis_bgm_as_numpy(prior_mus, np),
    prior_Sigmas = .praxis_bgm_as_numpy(prior_Sigmas, np),
    prior_weights = .praxis_bgm_as_numpy(prior_weights, np),
    init_mus = .praxis_bgm_as_numpy(init_mus, np),
    init_covs = .praxis_bgm_as_numpy(init_covs, np),
    init_pis = .praxis_bgm_as_numpy(init_pis, np),
    beta = as.numeric(beta),
    tol = as.numeric(tol),
    max_iters = as.integer(max_iters),
    verbose = isTRUE(verbose),
    sparse_A = .praxis_bgm_as_numpy(sparse_A, np),
    cluster_A = .praxis_bgm_as_numpy(cluster_A, np),
    freeze_A_zeros = isTRUE(freeze_A_zeros),
    prior_mus_variance = as.numeric(prior_mus_variance),
    num_samples = as.integer(num_samples),
    data_precision_int = if (is.null(data_precision_int)) NULL else as.integer(data_precision_int),
    likelihood_temp = as.numeric(likelihood_temp),
    rho_prec = as.numeric(rho_prec),
    rho_mu = as.numeric(rho_mu),
    elbo_eval_freq = as.integer(elbo_eval_freq)
  )

  model$fit(
    data_numpy,
    num_iters = as.integer(num_iters),
    batch_size = as.integer(batch_size),
    early_stop = isTRUE(early_stop),
    patience = as.integer(patience)
  )

  posteriors <- model$get_posteriors(data_numpy)
  prediction <- model$predict(data_numpy)

  posterior_mus <- .praxis_bgm_py_array_to_r(.praxis_bgm_extract_sequence(posteriors, 0L), np)
  posterior_covs <- .praxis_bgm_py_array_to_r(.praxis_bgm_extract_sequence(posteriors, 1L), np)
  posterior_pis <- .praxis_bgm_py_array_to_r(.praxis_bgm_extract_sequence(posteriors, 2L), np)
  responsibilities <- .praxis_bgm_py_array_to_r(.praxis_bgm_extract_sequence(posteriors, 3L), np)

  assignments_zero_based <- as.integer(
    .praxis_bgm_py_array_to_r(.praxis_bgm_extract_sequence(prediction, 0L), np)
  )
  learned_weights <- as.numeric(
    .praxis_bgm_py_array_to_r(.praxis_bgm_extract_sequence(prediction, 1L), np)
  )

  list(
    assignments = assignments_zero_based + 1L,
    assignments_zero_based = assignments_zero_based,
    learned_weights = learned_weights,
    posterior_mus = posterior_mus,
    posterior_covs = posterior_covs,
    posterior_pis = posterior_pis,
    responsibilities = responsibilities,
    model_summary = reticulate::py_to_r(model$get_model_summary()),
    elbo_history = as.numeric(reticulate::py_to_r(model$elbo_history)),
    model = model
  )
}

#' Run Bayes-factor feature selection for a fitted Praxis-BGM model
#'
#' The underlying BF scoring is delegated directly to the Python model, so the
#' selection logic is unchanged from Praxis-BGM itself.
#'
#' @param model A fitted Python Praxis-BGM model or the list returned by
#'   `praxis_bgm_fit()`.
#' @param data Numeric matrix or data frame with observations in rows.
#' @param top_n Positive integer number of top-ranked features to return.
#' @param visual Logical; whether to show the Python plotting output.
#'
#' @return A list containing the Bayes factor matrix, feature scores, 1-based
#'   and zero-based top-feature indices, and Jeffreys-scale classifications in
#'   both index conventions.
#' @export
praxis_bgm_bf_selection <- function(model, data, top_n = 20L, visual = FALSE) {
  .praxis_bgm_assert_python_modules()

  if (missing(model)) {
    stop("`model` is required and must be a fitted Praxis-BGM model.")
  }
  if (!is.numeric(top_n) || length(top_n) != 1L || is.na(top_n) ||
      top_n < 1 || top_n != as.integer(top_n)) {
    stop("`top_n` must be a positive integer.")
  }

  np <- reticulate::import("numpy", delay_load = TRUE, convert = FALSE)
  model_obj <- .praxis_bgm_resolve_model(model)
  data_matrix <- .praxis_bgm_validate_matrix(data)

  results <- model_obj$BF_selection(
    .praxis_bgm_as_numpy(data_matrix, np),
    top_n = as.integer(top_n),
    visual = isTRUE(visual)
  )

  classification_zero_based <- reticulate::py_to_r(.praxis_bgm_extract_sequence(results, 3L))
  top_features_zero_based <- as.integer(
    .praxis_bgm_py_array_to_r(.praxis_bgm_extract_sequence(results, 2L), np)
  )

  list(
    BF_matrix = .praxis_bgm_py_array_to_r(.praxis_bgm_extract_sequence(results, 0L), np),
    feature_scores = as.numeric(
      .praxis_bgm_py_array_to_r(.praxis_bgm_extract_sequence(results, 1L), np)
    ),
    top_features = top_features_zero_based + 1L,
    top_features_zero_based = top_features_zero_based,
    classification = .praxis_bgm_shift_index_list(classification_zero_based),
    classification_zero_based = classification_zero_based
  )
}

#' Simulate overlapping Gaussian-mixture samples
#'
#' Only the first `n_causal` features carry cluster-specific signal. The
#' remaining features are standard normal noise.
#'
#' @param n_components Positive integer number of clusters.
#' @param n_causal Positive integer number of causal features.
#' @param n_features Positive integer total number of features.
#' @param n_samples Positive integer total number of samples.
#' @param mean_shift Numeric scale for cluster-specific mean shifts.
#' @param random_seed Optional integer seed for reproducibility.
#'
#' @return A list with elements `samples`, `labels`, `full_means`, and `covs`.
#'   Labels are returned with zero-based indexing to mirror the Python tutorial.
#' @export
generate_overlapping_gmm_samples <- function(
  n_components,
  n_causal,
  n_features,
  n_samples,
  mean_shift = 0.4,
  random_seed = NULL
) {
  if (!is.null(random_seed)) {
    set.seed(random_seed)
  }
  if (n_components < 1L || n_causal < 1L || n_features < 1L || n_samples < 1L) {
    stop("`n_components`, `n_causal`, `n_features`, and `n_samples` must all be positive.")
  }
  if (n_causal > n_features) {
    stop("`n_causal` cannot exceed `n_features`.")
  }

  samples_per_component <- rep(n_samples %/% n_components, n_components)
  if (n_samples %% n_components > 0L) {
    samples_per_component[seq_len(n_samples %% n_components)] <-
      samples_per_component[seq_len(n_samples %% n_components)] + 1L
  }

  causal_means <- matrix(
    rnorm(n_components * n_causal, mean = 0, sd = mean_shift),
    nrow = n_components,
    ncol = n_causal
  )

  samples_list <- vector("list", n_components)
  labels_list <- vector("list", n_components)
  covs <- array(0, dim = c(n_components, n_features, n_features))

  for (k in seq_len(n_components)) {
    n_k <- samples_per_component[k]
    causal <- MASS::mvrnorm(n_k, causal_means[k, ], diag(n_causal))
    causal <- matrix(causal, nrow = n_k, ncol = n_causal)

    if (n_features > n_causal) {
      non_causal <- matrix(rnorm(n_k * (n_features - n_causal)), ncol = n_features - n_causal)
      samples_list[[k]] <- cbind(causal, non_causal)
    } else {
      samples_list[[k]] <- causal
    }

    labels_list[[k]] <- rep(k - 1L, n_k)
    covs[k, , ] <- diag(n_features)
  }

  full_means <- matrix(0, nrow = n_components, ncol = n_features)
  full_means[, seq_len(n_causal)] <- causal_means

  list(
    samples = do.call(rbind, samples_list),
    labels = unlist(labels_list, use.names = FALSE),
    full_means = full_means,
    covs = covs
  )
}

#' Randomly shift a subset of cluster means
#'
#' @param true_means Numeric matrix of true means with shape `K x P`.
#' @param shift_magnitude Numeric scale for Gaussian shifts.
#' @param percentage Fraction of features to shift in each cluster.
#' @param random_seed Optional integer seed for reproducibility.
#'
#' @return A matrix of shifted means with the same dimensions as `true_means`.
#' @export
randomly_shift_means <- function(true_means, shift_magnitude, percentage, random_seed = NULL) {
  if (!is.null(random_seed)) {
    set.seed(random_seed)
  }
  true_means <- as.matrix(true_means)
  if (!is.numeric(true_means) || length(dim(true_means)) != 2L) {
    stop("`true_means` must be a numeric matrix.")
  }
  if (!is.numeric(percentage) || length(percentage) != 1L || percentage < 0 || percentage > 1) {
    stop("`percentage` must be a single number between 0 and 1.")
  }
  if (percentage == 0) {
    return(true_means)
  }

  shifted <- true_means
  n_clusters <- nrow(shifted)
  n_features <- ncol(shifted)
  n_shift <- max(1L, min(n_features, as.integer(round(percentage * n_features))))

  for (k in seq_len(n_clusters)) {
    idx <- sample(seq_len(n_features), size = n_shift, replace = FALSE)
    shifted[k, idx] <- shifted[k, idx] + rnorm(n_shift, sd = shift_magnitude)
  }

  shifted
}

#' Align estimated and true means, then compute the Frobenius norm
#'
#' @param est_means Numeric matrix of estimated means.
#' @param true_means Numeric matrix of true means.
#'
#' @return Numeric Frobenius norm after Hungarian alignment.
#' @export
l2_norm_with_alignment <- function(est_means, true_means) {
  est_means <- as.matrix(est_means)
  true_means <- as.matrix(true_means)

  if (!is.numeric(est_means) || !is.numeric(true_means)) {
    stop("`est_means` and `true_means` must both be numeric matrices.")
  }
  if (!identical(dim(est_means), dim(true_means))) {
    stop("`est_means` and `true_means` must have the same dimensions.")
  }

  cost <- as.matrix(proxy::dist(est_means, true_means, method = "Euclidean"))
  assignment <- clue::solve_LSAP(cost)
  aligned_true <- true_means[assignment, , drop = FALSE]
  norm(est_means - aligned_true, type = "F")
}

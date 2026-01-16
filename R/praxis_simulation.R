#' Simulate overlapping GMM source data
#'
#' Generates clustered samples where only the first
#' \code{n_causal} features carry cluster-specific signal and the
#' remaining features are standard normal noise.
#'
#' @param n_components Integer number of clusters.
#' @param n_causal Integer number of causal features.
#' @param n_features Integer total number of features.
#' @param n_samples Integer total number of samples.
#' @param mean_shift Numeric scale for cluster-specific mean shifts.
#' @param random_seed Optional integer seed for reproducibility.
#'
#' @return A list with elements \code{samples}, \code{labels}, \code{full_means},
#'   and \code{covs} (NULL placeholder).
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

  base_mean <- rep(0, n_causal)
  causal_means <- vapply(
    seq_len(n_components),
    function(x) base_mean + rnorm(n_causal) * mean_shift,
    numeric(n_causal)
  )
  causal_means <- t(causal_means)

  samples_per_component <- n_samples %/% n_components
  samples_list <- vector("list", n_components)
  labels_list <- vector("list", n_components)

  for (k in seq_len(n_components)) {
    causal <- MASS::mvrnorm(samples_per_component, causal_means[k, ], diag(n_causal))
    non_causal <- matrix(
      rnorm(samples_per_component * (n_features - n_causal)),
      ncol = n_features - n_causal
    )
    samples_list[[k]] <- cbind(causal, non_causal)
    labels_list[[k]] <- rep(k - 1L, samples_per_component)
  }

  samples <- do.call(rbind, samples_list)
  labels <- unlist(labels_list)

  full_means <- matrix(0, nrow = n_components, ncol = n_features)
  full_means[, seq_len(n_causal)] <- causal_means

  list(samples = samples, labels = labels, full_means = full_means, covs = NULL)
}

#' Randomly shift cluster means to simulate domain shift
#'
#' @param true_means Matrix of true means (K x P).
#' @param shift_magnitude Numeric scale for the Gaussian shifts.
#' @param percentage Fraction of features to shift per cluster.
#' @param random_seed Integer seed for reproducibility.
#'
#' @return Matrix of shifted means with the same shape as \code{true_means}.
#' @export
randomly_shift_means <- function(true_means, shift_magnitude, percentage, random_seed) {
  set.seed(random_seed)
  shifted <- true_means
  n_clusters <- nrow(shifted)
  n_features <- ncol(shifted)
  n_shift <- as.integer(percentage * n_features)
  for (k in seq_len(n_clusters)) {
    idx <- sample(seq_len(n_features), size = n_shift, replace = FALSE)
    shifted[k, idx] <- shifted[k, idx] + rnorm(n_shift) * shift_magnitude
  }
  shifted
}

#' Align estimated means to true means and compute L2 distance
#'
#' @param est_means Matrix of estimated means.
#' @param true_means Matrix of true means.
#'
#' @return Numeric Frobenius norm after alignment.
#' @export
l2_norm_with_alignment <- function(est_means, true_means) {
  cost <- as.matrix(proxy::dist(est_means, true_means, method = "Euclidean"))
  assignment <- clue::solve_LSAP(cost)
  aligned_est <- est_means[seq_len(nrow(est_means)), , drop = FALSE]
  aligned_true <- true_means[assignment, , drop = FALSE]
  norm(aligned_est - aligned_true, type = "F")
}

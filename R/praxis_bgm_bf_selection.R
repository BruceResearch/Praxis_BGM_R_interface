library(reticulate)

use_condaenv("Praxis_env")

#' Perform Bayes factor feature selection for a fitted Praxis-BGM model
#'
#' @param model A fitted Praxis-BGM model or a list returned by `praxis_bgm_fit`.
#' @param data Numeric matrix or data frame with observations in rows.
#' @param top_n Integer number of top features to return.
#' @param visual Logical; whether to show diagnostic plots from the Python method.
#'
#' @return List containing the BF matrix, feature scores, top features, and
#'   Jeffreys scale classification buckets.
#' @export
praxis_bgm_bf_selection <- function(model, data, top_n = 20L, visual = FALSE) {
  if (!reticulate::py_module_available("praxis_bgm")) {
    stop("The 'praxis_bgm' Python module is not available. Install it before use.")
  }
  if (missing(model)) {
    stop("`model` is required and must be a fitted Praxis-BGM model.")
  }
  if (missing(data)) {
    stop("`data` is required and must be a numeric matrix or data frame.")
  }
  if (!is.numeric(top_n) || length(top_n) != 1 || top_n <= 0) {
    stop("`top_n` must be a positive integer.")
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

  np <- reticulate::import("numpy", delay_load = TRUE)

  model_obj <- if (is.list(model) && !is.null(model$model)) {
    model$model
  } else {
    model
  }

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

  if (!inherits(model_obj, "python.builtin.object")) {
    stop("`model` must be a Python Praxis-BGM object or a list from praxis_bgm_fit.")
  }

  results <- model_obj$BF_selection(
    as_numpy(data_matrix, np),
    top_n = as.integer(top_n),
    visual = visual
  )

  list(
    BF_matrix = reticulate::py_to_r(results[[1]]),
    feature_scores = reticulate::py_to_r(results[[2]]),
    top_features = reticulate::py_to_r(results[[3]]),
    classification = reticulate::py_to_r(results[[4]])
  )
}

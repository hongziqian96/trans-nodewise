###############################################################################
# Estimation functions for "Transfer Learning for Precision Matrix Estimation
# via Nodewise Regression": single-task, pooled, and Trans-Nodewise estimators
# (Lasso and SCAD penalties, HBIC-tuned), including the global- and
# oracle-diagonal variants of the Trans-Nodewise estimators.
###############################################################################

library(glmnet)      # Lasso
library(ncvreg)      # SCAD
library(doParallel)  # Parallel computing
library(foreach)     # Parallel loops
library(Matrix)      # Block-diagonal matrices

###############################################################################
# Section 0: Helper Functions
###############################################################################

# Column-center a single matrix. Used by center_stacked_domains.
center_columns <- function(X) {
  sweep(X, 2, colMeans(X), FUN = "-")
}

# Block-wise column-centering for stacked multi-domain data. Used by pooled and transfer estimators.
center_stacked_domains <- function(data, block_sizes = NULL) {
  if (is.null(block_sizes)) {
    return(center_columns(data))
  }

  stopifnot("Sum of block_sizes must equal the number of rows in data" =
              sum(block_sizes) == nrow(data))

  block_ends <- cumsum(block_sizes)
  block_starts <- c(1, head(block_ends, -1) + 1)

  centered_blocks <- vector("list", length(block_sizes))
  for (i in seq_along(block_sizes)) {
    centered_blocks[[i]] <- center_columns(
      data[block_starts[i]:block_ends[i], , drop = FALSE]
    )
  }

  do.call(rbind, centered_blocks)
}

# HBIC lambda selector for glmnet. Used by nodewise_lasso_hbic_parallel.
choose_lambda_hbic_glmnet <- function(x, y, p_hbic, alpha = 1,
                                      intercept = FALSE,
                                      standardize = FALSE,
                                      penalty.factor = NULL,
                                      lambda.min.ratio = 0.01) {
  n <- nrow(x)

  fit_args <- list(
    x = x,
    y = y,
    alpha = alpha,
    intercept = intercept,
    standardize = standardize,
    lambda.min.ratio = lambda.min.ratio
  )
  if (!is.null(penalty.factor)) {
    fit_args$penalty.factor <- penalty.factor
  }
  fit <- do.call(glmnet, fit_args)
  lambda_seq <- fit$lambda

  fitted_vals <- predict(fit, newx = x)
  residuals_mat <- matrix(y, nrow = n, ncol = length(lambda_seq)) - fitted_vals
  rss_vec <- colSums(residuals_mat^2)

  coef_mat <- coef(fit)
  df_vec <- apply(coef_mat[-1, , drop = FALSE], 2, function(b) sum(b != 0))

  hbic_vec <- log(rss_vec / n) + (log(p_hbic) * log(log(n)) / n) * df_vec
  valid_idx <- which(is.finite(hbic_vec) & is.finite(rss_vec) & (rss_vec > 0))

  if (!length(valid_idx)) {
    stop("HBIC failed: no valid glmnet path points available.")
  }

  lambda_seq[valid_idx[which.min(hbic_vec[valid_idx])]]
}

# HBIC lambda selector for ncvreg (SCAD). Used by nodewise_scad_hbic_parallel.
select_hbic_ncvreg_path <- function(x, y, p_hbic, penalty = "SCAD",
                                    penalty.factor = NULL,
                                    target_rows = NULL,
                                    coef_rows = NULL) {
  n_total <- nrow(x)
  if (is.null(target_rows)) {
    target_rows <- seq_len(n_total)
  }

  fit_args <- list(
    X = x,
    y = y,
    penalty = penalty
  )
  if (!is.null(penalty.factor)) {
    fit_args$penalty.factor <- penalty.factor
  }
  fit <- do.call(ncvreg, fit_args)
  lambda_seq <- fit$lambda

  fitted_vals <- predict(fit, X = x)
  residuals_mat <- matrix(y, nrow = n_total, ncol = length(lambda_seq)) - fitted_vals
  rss_vec <- colSums(residuals_mat[target_rows, , drop = FALSE]^2)

  coef_mat <- coef(fit)
  if (is.null(coef_rows)) {
    coef_rows <- 2:nrow(coef_mat)
  }
  df_vec <- apply(coef_mat[coef_rows, , drop = FALSE], 2, function(b) sum(b != 0))

  search_idx <- if (!is.null(fit$convex.min) && is.finite(fit$convex.min) && fit$convex.min >= 1) {
    seq_len(fit$convex.min)
  } else {
    seq_along(lambda_seq)
  }

  n_target <- length(target_rows)
  hbic_vec <- log(rss_vec / n_target) + (log(p_hbic) * log(log(n_target)) / n_target) * df_vec
  valid_idx <- search_idx[is.finite(hbic_vec[search_idx]) &
                            is.finite(rss_vec[search_idx]) &
                            (rss_vec[search_idx] > 0)]

  if (!length(valid_idx)) {
    stop("HBIC failed: no valid convex ncvreg path points available.")
  }

  best_idx <- valid_idx[which.min(hbic_vec[valid_idx])]

  list(
    fit = fit,
    index = best_idx,
    lambda = lambda_seq[best_idx],
    coef = coef_mat[, best_idx],
    fitted = fitted_vals[, best_idx]
  )
}

# HBIC lambda selector for ncvreg (SCAD). Used by nodewise_scad_hbic_parallel.
choose_lambda_hbic_ncvreg <- function(x, y, p_hbic, penalty = "SCAD",
                                      penalty.factor = NULL) {
  select_hbic_ncvreg_path(
    x, y,
    p_hbic = p_hbic,
    penalty = penalty,
    penalty.factor = penalty.factor
  )$lambda
}

# Target-aware HBIC lambda selector for glmnet. Used by trans_nodewise_lasso_hbic_column.
choose_lambda_hbic_glmnet_target <- function(x, y, target_rows, coef_rows,
                                             p_hbic, alpha = 1,
                                             intercept = FALSE,
                                             standardize = FALSE,
                                             penalty.factor = NULL,
                                             lambda.min.ratio = 0.01) {
  n_target <- length(target_rows)

  fit_args <- list(
    x = x,
    y = y,
    alpha = alpha,
    intercept = intercept,
    standardize = standardize,
    lambda.min.ratio = lambda.min.ratio
  )
  if (!is.null(penalty.factor)) {
    fit_args$penalty.factor <- penalty.factor
  }
  fit <- do.call(glmnet, fit_args)
  lambda_seq <- fit$lambda

  fitted_vals <- predict(fit, newx = x)
  residuals_mat <- matrix(y, nrow = nrow(x), ncol = length(lambda_seq)) - fitted_vals
  rss_vec <- colSums(residuals_mat[target_rows, , drop = FALSE]^2)

  coef_mat <- coef(fit)
  df_vec <- apply(coef_mat[coef_rows, , drop = FALSE], 2, function(b) sum(b != 0))

  hbic_vec <- log(rss_vec / n_target) + (log(p_hbic) * log(log(n_target)) / n_target) * df_vec
  valid_idx <- which(is.finite(hbic_vec) & is.finite(rss_vec) & (rss_vec > 0))

  if (!length(valid_idx)) {
    stop("Transfer HBIC failed: no valid glmnet path points available.")
  }

  lambda_seq[valid_idx[which.min(hbic_vec[valid_idx])]]
}

# Target-aware HBIC lambda selector for ncvreg (SCAD). Used by trans_nodewise_scad_hbic_column.
choose_lambda_hbic_ncvreg_target <- function(x, y, target_rows, coef_rows,
                                             p_hbic, penalty = "SCAD",
                                             penalty.factor = NULL) {
  select_hbic_ncvreg_path(
    x, y,
    p_hbic = p_hbic,
    penalty = penalty,
    penalty.factor = penalty.factor,
    target_rows = target_rows,
    coef_rows = coef_rows
  )$lambda
}

###############################################################################
# Section 1: Single-Task Estimators
###############################################################################

# Nodewise Lasso with HBIC tuning (parallel). Also used by pooled_nodewise_lasso_hbic_parallel.
nodewise_lasso_hbic_parallel <- function(data, alpha = 1, num_cores = 8,
                                         block_sizes = NULL) {
  X <- center_stacked_domains(data, block_sizes)
  n <- nrow(X)
  p <- ncol(X)

  Theta <- matrix(0, p, p)
  diag(Theta) <- 1

  cl <- makeCluster(num_cores)
  registerDoParallel(cl)

  results <- foreach(j = 1:p, .combine = rbind, .packages = "glmnet",
                     .export = "choose_lambda_hbic_glmnet") %dopar% {
    y <- X[, j]
    x <- X[, -j]

    optimal_lambda <- choose_lambda_hbic_glmnet(
      x, y,
      p_hbic = p,
      alpha = alpha,
      intercept = FALSE,
      standardize = FALSE
    )

    lasso.fit <- glmnet(
      x, y,
      alpha = alpha,
      lambda = optimal_lambda,
      intercept = FALSE,
      standardize = FALSE
    )
    betahat <- as.vector(coef(lasso.fit))[-1]

    s <- sum(betahat != 0)
    fitted_values <- as.numeric(predict(lasso.fit, newx = x))
    Theta_jj <- (n - s) / sum((y - fitted_values)^2)
    Theta_j <- -betahat * Theta_jj

    c(Theta_jj, Theta_j)
  }

  stopCluster(cl)

  for (j in 1:p) {
    Theta[j, j] <- results[j, 1]
    Theta[j, -j] <- results[j, -1]
  }

  Theta <- 0.5 * (Theta + t(Theta))
  return(Theta)
}

# Nodewise SCAD with HBIC tuning (parallel). Also used by pooled_nodewise_scad_hbic_parallel.
nodewise_scad_hbic_parallel <- function(data, num_cores = 8,
                                        block_sizes = NULL) {
  X <- center_stacked_domains(data, block_sizes)
  n <- nrow(X)
  p <- ncol(X)

  Theta <- matrix(0, p, p)
  diag(Theta) <- 1

  cl <- makeCluster(num_cores)
  registerDoParallel(cl)

  results <- foreach(j = 1:p, .combine = rbind, .packages = "ncvreg",
                     .export = "select_hbic_ncvreg_path") %dopar% {
    y <- X[, j]
    x <- X[, -j]

    scad_path <- select_hbic_ncvreg_path(
      x, y,
      p_hbic = p,
      penalty = "SCAD"
    )
    betahat <- as.vector(scad_path$coef)[-1]

    s <- sum(betahat != 0)
    fitted_values <- as.numeric(scad_path$fitted)
    Theta_jj <- (n - s) / sum((y - fitted_values)^2)
    Theta_j <- -betahat * Theta_jj

    c(Theta_jj, Theta_j)
  }

  stopCluster(cl)

  for (j in 1:p) {
    Theta[j, j] <- results[j, 1]
    Theta[j, -j] <- results[j, -1]
  }

  Theta <- 0.5 * (Theta + t(Theta))
  return(Theta)
}

###############################################################################
# Section 2: Pooled Estimators
###############################################################################

# Pooled Nodewise Lasso with HBIC. Wraps nodewise_lasso_hbic_parallel with block-wise centering.
pooled_nodewise_lasso_hbic_parallel <- function(data, n, alpha = 1,
                                                num_cores = 8) {
  nodewise_lasso_hbic_parallel(
    data,
    alpha = alpha,
    num_cores = num_cores,
    block_sizes = n
  )
}

# Pooled Nodewise SCAD with HBIC. Wraps nodewise_scad_hbic_parallel with block-wise centering.
pooled_nodewise_scad_hbic_parallel <- function(data, n,
                                               num_cores = 8) {
  nodewise_scad_hbic_parallel(
    data,
    num_cores = num_cores,
    block_sizes = n
  )
}

###############################################################################
# Section 3: Transfer Learning Estimators
###############################################################################

# Trans-Nodewise Lasso with HBIC tuning (parallel). Entry point for transfer Lasso HBIC.
trans_nodewise_lasso_hbic_parallel <- function(data, n, K, alpha = 1,
                                               p.fac = NULL, num_cores = 8) {
  X.centered <- center_stacked_domains(data, n)
  cl <- makeCluster(num_cores)
  registerDoParallel(cl)

  p <- ncol(X.centered)

  precision_matrix <- foreach(t = 1:p, .combine = rbind,
                              .packages = c("glmnet", "Matrix"),
                              .export = c("trans_nodewise_lasso_hbic_column",
                                          "choose_lambda_hbic_glmnet_target")) %dopar% {
    trans_nodewise_lasso_hbic_column(t, X.centered, n, K, alpha, p.fac)
  }

  stopCluster(cl)

  precision_matrix <- 0.5 * (precision_matrix + t(precision_matrix))
  return(unname(precision_matrix))
}

# Column-wise transfer Lasso with HBIC. Called by trans_nodewise_lasso_hbic_parallel.
trans_nodewise_lasso_hbic_column <- function(t, X.all, n, K, alpha,
                                             p.fac = NULL) {
  n_cum <- cumsum(n)
  nA <- sum(n[2:(K + 1)])
  p <- ncol(X.all)

  X <- rbind(X.all[(n[1] + 1):nrow(X.all), ], X.all[1:n[1], ])
  y <- X[, t]
  x <- X.all[1:n[1], -t, drop = FALSE]

  for (i in (K + 1):2) {
    x <- bdiag(X.all[(n_cum[i - 1] + 1):n_cum[i], -t], x)
  }
  x[1:nA, (K * (p - 1) + 1):(K * (p - 1) + p - 1)] <- X[1:nA, -t]

  p.fac.expand <- rep(1, (K + 1) * (p - 1))
  if (!is.null(p.fac)) {
    p.fac.expand <- rep(p.fac[c(2:(K + 1), 1)], each = p - 1)
  }

  target_rows <- (nA + 1):(nA + n[1])
  target_idx <- (K * (p - 1) + 2):(K * (p - 1) + p)

  lam <- choose_lambda_hbic_glmnet_target(
    x, y,
    target_rows = target_rows,
    coef_rows = target_idx,
    p_hbic = p,
    alpha = alpha,
    intercept = FALSE,
    standardize = FALSE,
    penalty.factor = p.fac.expand
  )

  lasso.fit <- glmnet(
    x, y,
    alpha = alpha,
    lambda = lam,
    intercept = FALSE,
    standardize = FALSE,
    penalty.factor = p.fac.expand
  )
  betahat <- as.vector(coef(lasso.fit))

  s <- sum(betahat[target_idx] != 0)
  fitted_values <- as.numeric(predict(lasso.fit, newx = x))

  temp <- rep(0, p)
  temp[t] <- (n[1] - s) / sum((y - fitted_values)[target_rows]^2)
  temp[-t] <- -betahat[target_idx] * temp[t]

  return(temp)
}

# Trans-Nodewise SCAD with HBIC tuning (parallel). Entry point for transfer SCAD HBIC.
trans_nodewise_scad_hbic_parallel <- function(data, n, K,
                                              p.fac = NULL, num_cores = 8) {
  X.centered <- center_stacked_domains(data, n)
  cl <- makeCluster(num_cores)
  registerDoParallel(cl)

  p <- ncol(X.centered)

  precision_matrix <- foreach(t = 1:p, .combine = rbind,
                              .packages = c("ncvreg", "Matrix"),
                              .export = c("trans_nodewise_scad_hbic_column",
                                          "select_hbic_ncvreg_path")) %dopar% {
    trans_nodewise_scad_hbic_column(t, X.centered, n, K, p.fac)
  }

  stopCluster(cl)

  precision_matrix <- 0.5 * (precision_matrix + t(precision_matrix))
  return(unname(precision_matrix))
}

# Column-wise transfer SCAD with HBIC. Called by trans_nodewise_scad_hbic_parallel.
trans_nodewise_scad_hbic_column <- function(t, X.all, n, K, p.fac = NULL) {
  n_cum <- cumsum(n)
  nA <- sum(n[2:(K + 1)])
  p <- ncol(X.all)

  X <- rbind(X.all[(n[1] + 1):nrow(X.all), ], X.all[1:n[1], ])
  y <- X[, t]
  x <- X.all[1:n[1], -t, drop = FALSE]

  for (i in (K + 1):2) {
    x <- bdiag(X.all[(n_cum[i - 1] + 1):n_cum[i], -t], x)
  }
  x[1:nA, (K * (p - 1) + 1):(K * (p - 1) + p - 1)] <- X[1:nA, -t]
  x <- as.matrix(x)

  p.fac.expand <- rep(1, (K + 1) * (p - 1))
  if (!is.null(p.fac)) {
    p.fac.expand <- rep(p.fac[c(2:(K + 1), 1)], each = p - 1)
  }

  target_rows <- (nA + 1):(nA + n[1])
  target_idx <- (K * (p - 1) + 2):(K * (p - 1) + p)

  scad_path <- select_hbic_ncvreg_path(
    x, y,
    p_hbic = p,
    penalty = "SCAD",
    penalty.factor = p.fac.expand,
    target_rows = target_rows,
    coef_rows = target_idx
  )
  betahat <- as.vector(scad_path$coef)

  s <- sum(betahat[target_idx] != 0)
  fitted_values <- as.numeric(scad_path$fitted)

  temp <- rep(0, p)
  temp[t] <- (n[1] - s) / sum((y - fitted_values)[target_rows]^2)
  temp[-t] <- -betahat[target_idx] * temp[t]

  return(temp)
}


###############################################################################
# Section 4: Global Diagonal Variants
###############################################################################

# Column-wise transfer Lasso HBIC with global diagonal estimate.
trans_nodewise_lasso_hbic_global_column <- function(t, X.all, n, K, alpha,
                                                    p.fac = NULL) {
  n_cum <- cumsum(n)
  nA <- sum(n[2:(K + 1)])
  p <- ncol(X.all)

  X <- rbind(X.all[(n[1] + 1):nrow(X.all), ], X.all[1:n[1], ])
  y <- X[, t]
  x <- X.all[1:n[1], -t, drop = FALSE]

  for (i in (K + 1):2) {
    x <- bdiag(X.all[(n_cum[i - 1] + 1):n_cum[i], -t], x)
  }
  x[1:nA, (K * (p - 1) + 1):(K * (p - 1) + p - 1)] <- X[1:nA, -t]

  p.fac.expand <- rep(1, (K + 1) * (p - 1))
  if (!is.null(p.fac)) {
    p.fac.expand <- rep(p.fac[c(2:(K + 1), 1)], each = p - 1)
  }

  target_rows <- (nA + 1):(nA + n[1])
  target_idx <- (K * (p - 1) + 2):(K * (p - 1) + p)

  lam <- choose_lambda_hbic_glmnet_target(
    x, y,
    target_rows = target_rows,
    coef_rows = target_idx,
    p_hbic = p,
    alpha = alpha,
    intercept = FALSE,
    standardize = FALSE,
    penalty.factor = p.fac.expand
  )

  lasso.fit <- glmnet(
    x, y,
    alpha = alpha,
    lambda = lam,
    intercept = FALSE,
    standardize = FALSE,
    penalty.factor = p.fac.expand
  )
  betahat <- as.vector(coef(lasso.fit))

  # Global: count all nonzero coefficients (skip intercept at index 1)
  s <- sum(betahat[-1] != 0)
  fitted_values <- as.numeric(predict(lasso.fit, newx = x))

  # Global: use all residuals and total sample size
  temp <- rep(0, p)
  temp[t] <- (sum(n) - s) / sum((y - fitted_values)^2)
  temp[-t] <- -betahat[target_idx] * temp[t]

  return(temp)
}

# Parallel wrapper for global transfer Lasso HBIC.
trans_nodewise_lasso_hbic_global_parallel <- function(data, n, K, alpha = 1,
                                                      p.fac = NULL,
                                                      num_cores = 8) {
  X.centered <- center_stacked_domains(data, n)
  cl <- makeCluster(num_cores)
  registerDoParallel(cl)

  p <- ncol(X.centered)

  precision_matrix <- foreach(t = 1:p, .combine = rbind,
                              .packages = c("glmnet", "Matrix"),
                              .export = c("trans_nodewise_lasso_hbic_global_column",
                                          "choose_lambda_hbic_glmnet_target")) %dopar% {
    trans_nodewise_lasso_hbic_global_column(t, X.centered, n, K, alpha, p.fac)
  }

  stopCluster(cl)

  precision_matrix <- 0.5 * (precision_matrix + t(precision_matrix))
  return(unname(precision_matrix))
}

# Column-wise transfer SCAD HBIC with global diagonal estimate.
trans_nodewise_scad_hbic_global_column <- function(t, X.all, n, K,
                                                   p.fac = NULL) {
  n_cum <- cumsum(n)
  nA <- sum(n[2:(K + 1)])
  p <- ncol(X.all)

  X <- rbind(X.all[(n[1] + 1):nrow(X.all), ], X.all[1:n[1], ])
  y <- X[, t]
  x <- X.all[1:n[1], -t, drop = FALSE]

  for (i in (K + 1):2) {
    x <- bdiag(X.all[(n_cum[i - 1] + 1):n_cum[i], -t], x)
  }
  x[1:nA, (K * (p - 1) + 1):(K * (p - 1) + p - 1)] <- X[1:nA, -t]
  x <- as.matrix(x)

  p.fac.expand <- rep(1, (K + 1) * (p - 1))
  if (!is.null(p.fac)) {
    p.fac.expand <- rep(p.fac[c(2:(K + 1), 1)], each = p - 1)
  }

  target_rows <- (nA + 1):(nA + n[1])
  target_idx <- (K * (p - 1) + 2):(K * (p - 1) + p)

  scad_path <- select_hbic_ncvreg_path(
    x, y,
    p_hbic = p,
    penalty = "SCAD",
    penalty.factor = p.fac.expand,
    target_rows = target_rows,
    coef_rows = target_idx
  )
  betahat <- as.vector(scad_path$coef)

  # Global: count all nonzero coefficients (skip intercept at index 1)
  s <- sum(betahat[-1] != 0)
  fitted_values <- as.numeric(scad_path$fitted)

  # Global: use all residuals and total sample size
  temp <- rep(0, p)
  temp[t] <- (sum(n) - s) / sum((y - fitted_values)^2)
  temp[-t] <- -betahat[target_idx] * temp[t]

  return(temp)
}

# Parallel wrapper for global transfer SCAD HBIC.
trans_nodewise_scad_hbic_global_parallel <- function(data, n, K,
                                                     p.fac = NULL,
                                                     num_cores = 8) {
  X.centered <- center_stacked_domains(data, n)
  cl <- makeCluster(num_cores)
  registerDoParallel(cl)

  p <- ncol(X.centered)

  precision_matrix <- foreach(t = 1:p, .combine = rbind,
                              .packages = c("ncvreg", "Matrix"),
                              .export = c("trans_nodewise_scad_hbic_global_column",
                                          "select_hbic_ncvreg_path")) %dopar% {
    trans_nodewise_scad_hbic_global_column(t, X.centered, n, K, p.fac)
  }

  stopCluster(cl)

  precision_matrix <- 0.5 * (precision_matrix + t(precision_matrix))
  return(unname(precision_matrix))
}

###############################################################################
# Section 5: Oracle Diagonal Variants
###############################################################################

# Column-wise transfer Lasso HBIC with oracle diagonal (true Omega_target[t,t]).
trans_nodewise_lasso_hbic_oracle_column <- function(t, X.all, n, K, alpha,
                                                    Omega_target,
                                                    p.fac = NULL) {
  n_cum <- cumsum(n)
  nA <- sum(n[2:(K + 1)])
  p <- ncol(X.all)

  X <- rbind(X.all[(n[1] + 1):nrow(X.all), ], X.all[1:n[1], ])
  y <- X[, t]
  x <- X.all[1:n[1], -t, drop = FALSE]

  for (i in (K + 1):2) {
    x <- bdiag(X.all[(n_cum[i - 1] + 1):n_cum[i], -t], x)
  }
  x[1:nA, (K * (p - 1) + 1):(K * (p - 1) + p - 1)] <- X[1:nA, -t]

  p.fac.expand <- rep(1, (K + 1) * (p - 1))
  if (!is.null(p.fac)) {
    p.fac.expand <- rep(p.fac[c(2:(K + 1), 1)], each = p - 1)
  }

  target_rows <- (nA + 1):(nA + n[1])
  target_idx <- (K * (p - 1) + 2):(K * (p - 1) + p)

  lam <- choose_lambda_hbic_glmnet_target(
    x, y,
    target_rows = target_rows,
    coef_rows = target_idx,
    p_hbic = p,
    alpha = alpha,
    intercept = FALSE,
    standardize = FALSE,
    penalty.factor = p.fac.expand
  )

  lasso.fit <- glmnet(
    x, y,
    alpha = alpha,
    lambda = lam,
    intercept = FALSE,
    standardize = FALSE,
    penalty.factor = p.fac.expand
  )
  betahat <- as.vector(coef(lasso.fit))

  # Oracle: use true diagonal
  temp <- rep(0, p)
  temp[t] <- Omega_target[t, t]
  temp[-t] <- -betahat[target_idx] * temp[t]

  return(temp)
}

# Parallel wrapper for oracle transfer Lasso HBIC.
trans_nodewise_lasso_hbic_oracle_parallel <- function(data, n, K,
                                                      Omega_target,
                                                      alpha = 1,
                                                      p.fac = NULL,
                                                      num_cores = 8) {
  X.centered <- center_stacked_domains(data, n)
  cl <- makeCluster(num_cores)
  registerDoParallel(cl)

  p <- ncol(X.centered)

  precision_matrix <- foreach(t = 1:p, .combine = rbind,
                              .packages = c("glmnet", "Matrix"),
                              .export = c("trans_nodewise_lasso_hbic_oracle_column",
                                          "choose_lambda_hbic_glmnet_target")) %dopar% {
    trans_nodewise_lasso_hbic_oracle_column(t, X.centered, n, K, alpha,
                                            Omega_target, p.fac)
  }

  stopCluster(cl)

  precision_matrix <- 0.5 * (precision_matrix + t(precision_matrix))
  return(unname(precision_matrix))
}

# Column-wise transfer SCAD HBIC with oracle diagonal (true Omega_target[t,t]).
trans_nodewise_scad_hbic_oracle_column <- function(t, X.all, n, K,
                                                   Omega_target,
                                                   p.fac = NULL) {
  n_cum <- cumsum(n)
  nA <- sum(n[2:(K + 1)])
  p <- ncol(X.all)

  X <- rbind(X.all[(n[1] + 1):nrow(X.all), ], X.all[1:n[1], ])
  y <- X[, t]
  x <- X.all[1:n[1], -t, drop = FALSE]

  for (i in (K + 1):2) {
    x <- bdiag(X.all[(n_cum[i - 1] + 1):n_cum[i], -t], x)
  }
  x[1:nA, (K * (p - 1) + 1):(K * (p - 1) + p - 1)] <- X[1:nA, -t]
  x <- as.matrix(x)

  p.fac.expand <- rep(1, (K + 1) * (p - 1))
  if (!is.null(p.fac)) {
    p.fac.expand <- rep(p.fac[c(2:(K + 1), 1)], each = p - 1)
  }

  target_rows <- (nA + 1):(nA + n[1])
  target_idx <- (K * (p - 1) + 2):(K * (p - 1) + p)

  scad_path <- select_hbic_ncvreg_path(
    x, y,
    p_hbic = p,
    penalty = "SCAD",
    penalty.factor = p.fac.expand,
    target_rows = target_rows,
    coef_rows = target_idx
  )
  betahat <- as.vector(scad_path$coef)

  # Oracle: use true diagonal
  temp <- rep(0, p)
  temp[t] <- Omega_target[t, t]
  temp[-t] <- -betahat[target_idx] * temp[t]

  return(temp)
}

# Parallel wrapper for oracle transfer SCAD HBIC.
trans_nodewise_scad_hbic_oracle_parallel <- function(data, n, K,
                                                     Omega_target,
                                                     p.fac = NULL,
                                                     num_cores = 8) {
  X.centered <- center_stacked_domains(data, n)
  cl <- makeCluster(num_cores)
  registerDoParallel(cl)

  p <- ncol(X.centered)

  precision_matrix <- foreach(t = 1:p, .combine = rbind,
                              .packages = c("ncvreg", "Matrix"),
                              .export = c("trans_nodewise_scad_hbic_oracle_column",
                                          "select_hbic_ncvreg_path")) %dopar% {
    trans_nodewise_scad_hbic_oracle_column(t, X.centered, n, K,
                                           Omega_target, p.fac)
  }

  stopCluster(cl)

  precision_matrix <- 0.5 * (precision_matrix + t(precision_matrix))
  return(unname(precision_matrix))
}

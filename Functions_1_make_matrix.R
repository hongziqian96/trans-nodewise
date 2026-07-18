##### Target precision matrix: AR(1) structure

make_Sigma_AR1 <- function(p, rho = 0.5) {
  idx <- 1:p
  Sigma <- outer(idx, idx, FUN = function(i, j) rho^(abs(i - j)))
  return(Sigma)
}

make_Omega_AR1 <- function(p, rho = 0.5) {
  theta_0 <- 1 - rho^2
  theta_1 <- (1 + rho^2) / theta_0
  theta_2 <- -rho / theta_0
  theta_3 <- -rho^2 / theta_0

  A1 <- diag(p)
  A2 <- matrix(0, p, p)
  A2[row(A2) == col(A2) + 1] <- 1
  A2[row(A2) == col(A2) - 1] <- 1
  A3 <- matrix(0, p, p)
  A3[1, 1] <- 1
  A3[p, p] <- 1

  R_inv <- theta_1 * A1 + theta_2 * A2 + theta_3 * A3
  return(R_inv)
}

##### Source precision matrix: rank-1 perturbation
# Omega^(k) = Omega^(0) + a^(k) (a^(k))^T
# where ||Delta||_1 = ||a||_1 * ||a||_infty
#
# Both options use a random support S_k of size m = floor(0.1 * p),
# sampled uniformly without replacement from {1, ..., p}.

# Option 1: Calibrated uniform perturbation on random support
# For i in S_k: a_i ~ U(-sqrt(2h(m+2)/(m(m+1))), sqrt(2h(m+2)/(m(m+1))))
# This guarantees E[||Delta||_1] = h.
make_source_perturbation_uniform <- function(p, h, m = floor(0.1 * p)) {
  a <- rep(0, p)
  support <- sample(1:p, m)
  bound <- sqrt(2 * h * (m + 2) / (m * (m + 1)))
  a[support] <- runif(m, -bound, bound)
  return(a)
}

# Option 2: Rademacher perturbation on random support
# For i in S_k: a_i = xi_i * sqrt(h/m), xi_i ~ Rademacher
# This guarantees ||Delta||_1 = h exactly.
make_source_perturbation_rademacher <- function(p, h, m = floor(0.1 * p)) {
  a <- rep(0, p)
  support <- sample(1:p, m)
  signs <- sample(c(-1, 1), m, replace = TRUE)
  a[support] <- signs * sqrt(h / m)
  return(a)
}

# Generate source precision matrix from target + perturbation vector
# Returns list with Omega_source, Sigma_source, perturbation vector a, and Delta
make_source_precision <- function(Omega_target, Sigma_target, a) {
  p <- nrow(Omega_target)
  stopifnot("Dimensions of Omega_target, Sigma_target, and a must match" =
              ncol(Omega_target) == p && nrow(Sigma_target) == p &&
              ncol(Sigma_target) == p && length(a) == p)

  # Rank-1 perturbation: Omega^(k) = Omega^(0) + a a^T
  Delta <- outer(a, a)
  Omega_source <- Omega_target + Delta

  # Source covariance via Sherman-Morrison formula
  Sa <- Sigma_target %*% a
  denom <- 1 + sum(a * Sa)
  Sigma_source <- Sigma_target - (Sa %*% t(Sa)) / denom

  return(list(
    Omega  = Omega_source,
    Sigma  = Sigma_source,
    a      = a,
    Delta  = Delta
  ))
}

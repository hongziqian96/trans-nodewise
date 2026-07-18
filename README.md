# Trans-Nodewise: estimation functions

Core R functions for the paper *"Transfer Learning for Precision Matrix Estimation via Nodewise Regression"*. The proposed **Trans-Nodewise** estimators borrow strength from auxiliary source datasets to estimate a high-dimensional target precision matrix column by column: for each node j, the target and source nodewise regressions are stacked into one penalized least-squares problem (Lasso or SCAD penalty) that models the source–target discrepancy as sparse bias vectors, and the fitted coefficients and error variance are mapped back to the j-th column of the precision matrix. All tuning is by the high-dimensional BIC (HBIC).

## Files

- `Functions_1_make_matrix.R` — data generation: AR(1) target covariance/precision (`make_Sigma_AR1`, `make_Omega_AR1`), sparse perturbation vectors (`make_source_perturbation_uniform` / `_rademacher`), and rank-1 perturbed source models (`make_source_precision`).
- `Functions_2_estimation.R` — all estimators used in the paper: target-only, pooled, and Trans-Nodewise (local, global, and oracle diagonal), each with Lasso and SCAD penalties, plus the HBIC selectors and the parallel backend.
- `Functions_3_evaluation.R` — evaluation: `spectral_error`, `frobenius_error`, `matrix_l1_norm`, and `edge_metrics` (TPR and FDR of edge recovery).

## Requirements

```r
install.packages(c("MASS", "glmnet", "ncvreg", "doParallel", "foreach", "Matrix"))
```

## Usage

Every estimator takes a row-stacked data matrix with the **target block first**, followed by the K source blocks, together with `n`, the vector of block sample sizes `c(n_target, n_source_1, ..., n_source_K)`. Each block is mean-centered internally, and each function returns a p × p precision matrix estimate computed column by column in parallel (`num_cores`).

```r
source("Functions_1_make_matrix.R")
source("Functions_2_estimation.R")
source("Functions_3_evaluation.R")

set.seed(1)
p <- 50; n_t <- 50; n_s <- 400; K <- 2

# Target: AR(1) model with tridiagonal precision matrix
Sigma0 <- make_Sigma_AR1(p, rho = 0.6)
Omega0 <- make_Omega_AR1(p, rho = 0.6)
X_target <- MASS::mvrnorm(n_t, rep(0, p), Sigma0)

# Sources: rank-1 perturbations Omega^(k) = Omega^(0) + a a'
X_sources <- lapply(1:K, function(k) {
  a   <- make_source_perturbation_uniform(p, h = 5)  # or _rademacher
  src <- make_source_precision(Omega0, Sigma0, a)    # list(Omega, Sigma, a, Delta)
  MASS::mvrnorm(n_s, rep(0, p), src$Sigma)
})

# Stack target first, then sources; record block sizes
X_all <- rbind(X_target, do.call(rbind, X_sources))
n_vec <- c(n_t, rep(n_s, K))

# Estimation (Lasso shown; swap "lasso" for "scad" for the SCAD versions)
Omega_single <- nodewise_lasso_hbic_parallel(X_target, num_cores = 8)
Omega_pooled <- pooled_nodewise_lasso_hbic_parallel(X_all, n_vec, num_cores = 8)
Omega_trans  <- trans_nodewise_lasso_hbic_global_parallel(X_all, n_vec, K, num_cores = 8)

# Evaluation against the true target precision matrix
spectral_error(Omega_trans, Omega0)
frobenius_error(Omega_trans, Omega0)
edge_metrics(Omega_trans, Omega0)  # c(TPR, FDR)
```

## Estimator reference

| Function | Estimator |
|----------|-----------|
| `nodewise_{lasso,scad}_hbic_parallel(data, ...)` | Single-task, target data only |
| `pooled_nodewise_{lasso,scad}_hbic_parallel(data, n, ...)` | Naive pooling of target and sources |
| `trans_nodewise_{lasso,scad}_hbic_parallel(data, n, K, ...)` | Trans-Nodewise, local diagonal |
| `trans_nodewise_{lasso,scad}_hbic_global_parallel(data, n, K, ...)` | Trans-Nodewise, global (pooled-residual) diagonal — the paper's default |
| `trans_nodewise_{lasso,scad}_hbic_oracle_parallel(data, n, K, Omega_target, ...)` | Oracle-diagonal benchmark (requires the true precision matrix) |

The returned estimates are column-wise and therefore not exactly symmetric; symmetrize by `(Omega_hat + t(Omega_hat)) / 2` and, if a positive semidefinite estimate is needed, truncate negative eigenvalues at zero.
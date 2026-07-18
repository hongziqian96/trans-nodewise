##### Evaluation metrics

# Spectral norm error (primary metric for this simulation)
spectral_error <- function(Theta_hat, Omega) {
  norm(Theta_hat - Omega, "2")
}

# Frobenius norm error
frobenius_error <- function(Theta_hat, Omega) {
  norm(Theta_hat - Omega, "F")
}

# Matrix L1 norm (max absolute column sum)
matrix_l1_norm <- function(A) {
  max(colSums(abs(A)))
}

# Edge-support recovery metrics: TPR and FDR
edge_metrics <- function(est, truth) {
  mask_true <- (truth != 0) & upper.tri(truth)
  mask_est  <- (est != 0) & upper.tri(est)
  tp <- sum(mask_est & mask_true)
  fp <- sum(mask_est & !mask_true)
  fn <- sum(!mask_est & mask_true)
  tpr <- if (tp + fn > 0) tp / (tp + fn) else NA_real_
  fdr <- if (tp + fp > 0) fp / (tp + fp) else NA_real_
  c(TPR = tpr, FDR = fdr)
}

# Edge-support recovery metrics: TPR and FPR (unused; kept for reference)
# TPR = TP / (TP + FN): fraction of true edges correctly recovered
# FPR = FP / (FP + TN): fraction of true non-edges incorrectly flagged
# edge_metrics <- function(est, truth) {
#   upper <- upper.tri(truth)
#   mask_true <- (truth != 0) & upper
#   mask_neg  <- (truth == 0) & upper
#   mask_est  <- (est != 0) & upper
#   tp <- sum(mask_est & mask_true)
#   fp <- sum(mask_est & mask_neg)
#   fn <- sum(!mask_est & mask_true)
#   tn <- sum(!mask_est & mask_neg)
#   tpr <- if (tp + fn > 0) tp / (tp + fn) else NA_real_
#   fpr <- if (fp + tn > 0) fp / (fp + tn) else NA_real_
#   c(TPR = tpr, FPR = fpr)
# }

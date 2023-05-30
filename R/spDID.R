# %%
ipwDID = function(X, D, Y1, Y0) {
  #' Abadie (2005) semiparametric difference in differences estimator for the
  #' ATT fit using LASSO
  #' @param X NxK matrix of covariates for propensity score
  #' @param D N-vector of treatment assignments (treatment only applies in period 2)
  #' @param Y1 N-vector of outcomes in the second period
  #' @param Y0 N-vector of outcomes in the first period
  #' @return ATT estimate
  #' @references Abadie(2005)
  #' @export
  # fit propensity score
  fit = glmnet::cv.glmnet(X, D, family = "binomial", alpha = 1, keep = TRUE)
  ehat = predict(fit, newx = X, type = "response", s = fit$lambda.min)
  # IPW estimate
  (1 / mean(D)) * mean((D - ehat) / (1 - ehat) * (Y1 - Y0))
}

# %%
aipwDID = function(X, D, Y1, Y0) {
  #' Chang (2020) Double-Robust semiparametric difference in differences
  #' estimator for the ATT fit using LASSO
  #' @param X NxK matrix of covariates for propensity score
  #' @param D N-vector of treatment assignments (treatment only applies in period 2)
  #' @param Y1 N-vector of outcomes in the second period
  #' @param Y0 N-vector of outcomes in the first period
  #' @return ATT estimate
  #' @references  Chang(2020), Zhao and Sant'Anna (2021)
  #' @export
  k_folds = floor(max(3, min(10, length(D) / 4)))
  # fold ID for cross-validation; balance treatment assignments
  foldid = sample(rep(seq(k_folds), length = length(D)))
  # ps model
  psfit = glmnet::cv.glmnet(X, D, family = "binomial", alpha = 1, foldid = foldid, keep = TRUE)
  # trend model
  index = which(D == 0)
  y = Y1[index] - Y0[index]
  ofit = glmnet::cv.glmnet(X[index, ], y, alpha = 1, keep = TRUE, foldid = foldid[index])
  ehat = predict(psfit, newx = X, type = "response", s = psfit$lambda.min)
  mhat = predict(ofit, newx = X, type = "response", s = ofit$lambda.min)
  mean(
    (Y1 - Y0) / mean(D) * (D - ehat) / (1 - ehat) - (D - ehat) / mean(D) / (1 - ehat) * mhat
  )
}

# %%
DID = function(D, Y1, Y0) {
  #' Vanilla Difference in Differences estimation
  #' @param D N-vector of treatment assignments (treatment only applies in period 2)
  #' @param Y1 N-vector of outcomes in the second period
  #' @param Y0 N-vector of outcomes in the first period
  #' @return ATT estimate
  #' @export
  #' @references Ashenfelter (1977), Snow(18xx)
  (mean(Y1[D == 1]) - mean(Y0[D == 1])) - (mean(Y1[D == 0]) - mean(Y0[D == 0]))
}

# %%
omDID = function(X, D, Y1, Y0) {
  #' outcome model for the ATT fit using LASSO
  #' @param X NxK matrix of covariates for propensity score
  #' @param D N-vector of treatment assignments (treatment only applies in period 2)
  #' @param Y1 N-vector of outcomes in the second period
  #' @param Y0 N-vector of outcomes in the first period
  #' @return ATT estimate
  #' @references Heckman et al (1997), Abadie(2005)
  #' @export
  d0 = which(D == 0); d1 = which(D == 1);
  m1 = glmnet::cv.glmnet(y = Y1[d1], x = X[d1, ], alpha = 1, keep = TRUE)
  m2 = glmnet::cv.glmnet(y = Y1[d0], x = X[d0, ], alpha = 1, keep = TRUE)
  m3 = glmnet::cv.glmnet(y = Y0[d1], x = X[d1, ], alpha = 1, keep = TRUE)
  m4 = glmnet::cv.glmnet(y = Y0[d0], x = X[d0, ], alpha = 1, keep = TRUE)

  m1hat = predict(m1, newx = X, s = m1$lambda.min)
  m2hat = predict(m2, newx = X, s = m2$lambda.min)
  m3hat = predict(m3, newx = X, s = m3$lambda.min)
  m4hat = predict(m4, newx = X, s = m4$lambda.min)
  mean((m1hat - m2hat) - (m3hat - m4hat))
}

# %%
balDID = function(X, W, Y1, Y0) {
  #' semiparametric difference in differences estimator for the ATT (balancing weights)
  #' @param X NxK matrix of covariates for propensity score
  #' @param W N-vector of treatment assignments (treatment only applies in period 2)
  #' @param Y1 N-vector of outcomes in the second period
  #' @param Y0 N-vector of outcomes in the first period
  #' @return ATT estimate
  #' @references Abadie(2005)
  #' @export
  ω = ebal::ebalance(W, X, method = "AutoDiff")$w
  Δ = Y1 - Y0
  mean(Δ[W == 1]) - weighted.mean(Δ[W == 0], ω)
}


# %%
abalDID = function(X, D, Y1, Y0) {
  #' Augmented Balancing semiparametric difference in differences
  #' estimator for the ATT fit using LASSO
  #' @param X NxK matrix of covariates for propensity score
  #' @param D N-vector of treatment assignments (treatment only applies in period 2)
  #' @param Y1 N-vector of outcomes in the second period
  #' @param Y0 N-vector of outcomes in the first period
  #' @return ATT estimate
  #' @references  Chang(2020), Zhao and Sant'Anna (2021), Lal(2023)
  #' @export
  ρ = 1 / mean(D)
  ω = ebal::ebalance(D, X, method = "AutoDiff")$w
  index = which(D == 0); y = Y1[index] - Y0[index]
  ofit = glmnet::cv.glmnet(X[index, ], y, alpha = 1, keep = TRUE)
  μhat = predict(ofit, newx = X, type = "response", s = ofit$lambda.min)
  Δ_i = Y1 - Y0 - μhat
  mean(Δ_i[D == 1]) - sum(Δ_i[D == 0] * ω) / ρ
}

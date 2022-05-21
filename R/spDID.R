# %%
fitGet = \(m){
    m$fit.preval[,
      !is.na(colSums(m$fit.preval))][,
      m$lambda[!is.na(colSums(m$fit.preval))] == m$lambda.min]
}

# %%
ipwDID = function(X, D, Y1, Y0, xfit = T){
  #' Abadie (2005) semiparametric difference in differences estimator for the
  #' ATT fit using LASSO
  #' @param X NxK matrix of covariates for propensity score
  #' @param D N-vector of treatment assignments (treatment only applies in period 2)
  #' @param Y1 N-vector of outcomes in the second period
  #' @param Y0 N-vector of outcomes in the first period
  #' @return ATT estimate
  #' @references Abadie(2005)
  #' @export
  fit = glmnet::cv.glmnet(X, D, family="binomial", alpha=1, keep = TRUE)
  if(!xfit){
      ehat = predict(fit, newx = X, type = "response", s = fit$lambda.min)
  } else{ # cross fit predictions
      ehat = fitGet(fit) %>% expit()
  }
  # IPW estimate
  (1/mean(D)) * mean((D-ehat)/(1-ehat) * (Y1-Y0))
}

aipwDID = function(X, D, Y1, Y0, xfit = T){
  #' Chang (2020) Double-Robust semiparametric difference in differences
  #' estimator for the ATT fit using LASSO
  #' @param X NxK matrix of covariates for propensity score
  #' @param D N-vector of treatment assignments (treatment only applies in period 2)
  #' @param Y1 N-vector of outcomes in the second period
  #' @param Y0 N-vector of outcomes in the first period
  #' @return ATT estimate
  #' @references Abadie(2005), Chang(2020), Zhao and Sant'Anna (2021)
  #' @export
  # ps model
  psfit = glmnet::cv.glmnet(X, D, family="binomial", alpha=1, keep = TRUE)
  # trend model
  index = which(D==0)
  y = Y1[index]-Y0[index]
  ofit  = glmnet::cv.glmnet(X[index, ], y, alpha=1, keep = TRUE)
  if(!xfit){
      ehat = predict(psfit, newx = X, type = "response", s = psfit$lambda.min)
      mhat = predict(ofit,  newx = X, type = "response", s = ofit$lambda.min)
  } else{ # cross fit predictions
      ehat = fitGet(psfit) %>% expit()
      mhat = predict(ofit, newx = X, type = "response", s = ofit$lambda.min)
      # fill in obs that model was trained on using oob predictions
      mhat[index] = fitGet(ofit)
  }
  mean(
    (Y1-Y0)/mean(D) * (D-ehat)/(1-ehat) - (D-ehat)/mean(D)/(1-ehat) * mhat
  )
}

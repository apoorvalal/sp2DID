# %% get out-of-sample (linear) predictions from
# glmnet model fit on gaussian or binomial family ! does not work for
# multinomial logit because predicted values are in a matrix

fitGet = \(m){
  m$fit.preval[, !is.na(colSums(m$fit.preval))][,
    m$lambda[!is.na(colSums(m$fit.preval))] == m$lambda.min
  ]
}

# %% make probability from linear
expit = \(x) exp(x) / (1 + exp(x))

# %%

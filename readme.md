Semiparametric Difference in Differences Estimators using the LASSO
================
Apoorva Lal

The Abadie (2005) semiparametric IPW estimator for the 2-period ATT is

![\hat{\tau}^{\text{IPW}} = \frac{1}{N} \sum_i \frac{Y_i(1) - Y_i(0)}{P(D=1)} \frac{D - \hat{e}(X_i)}{1 - \hat{e}(X_i)}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Chat%7B%5Ctau%7D%5E%7B%5Ctext%7BIPW%7D%7D%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_i%20%5Cfrac%7BY_i%281%29%20-%20Y_i%280%29%7D%7BP%28D%3D1%29%7D%20%5Cfrac%7BD%20-%20%5Chat%7Be%7D%28X_i%29%7D%7B1%20-%20%5Chat%7Be%7D%28X_i%29%7D "\hat{\tau}^{\text{IPW}} = \frac{1}{N} \sum_i \frac{Y_i(1) - Y_i(0)}{P(D=1)} \frac{D - \hat{e}(X_i)}{1 - \hat{e}(X_i)}")

The Chang (2020) double-robust AIPW estimator is

![\hat{\tau}^{\text{AIPW}} = \frac{1}{N} \sum_i \frac{Y_i(1) - Y_i(0)}{P(D=1)} \frac{D - \hat{e}(X_i)}{1 - \hat{e}(X_i)} - \frac{D - \hat{e}(X_i)}{P(D = 1) (1- \hat{e}(X_i) )} \hat{\mathcal{l}}(X_i, D = 0)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Chat%7B%5Ctau%7D%5E%7B%5Ctext%7BAIPW%7D%7D%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_i%20%5Cfrac%7BY_i%281%29%20-%20Y_i%280%29%7D%7BP%28D%3D1%29%7D%20%5Cfrac%7BD%20-%20%5Chat%7Be%7D%28X_i%29%7D%7B1%20-%20%5Chat%7Be%7D%28X_i%29%7D%20-%20%5Cfrac%7BD%20-%20%5Chat%7Be%7D%28X_i%29%7D%7BP%28D%20%3D%201%29%20%281-%20%5Chat%7Be%7D%28X_i%29%20%29%7D%20%5Chat%7B%5Cmathcal%7Bl%7D%7D%28X_i%2C%20D%20%3D%200%29 "\hat{\tau}^{\text{AIPW}} = \frac{1}{N} \sum_i \frac{Y_i(1) - Y_i(0)}{P(D=1)} \frac{D - \hat{e}(X_i)}{1 - \hat{e}(X_i)} - \frac{D - \hat{e}(X_i)}{P(D = 1) (1- \hat{e}(X_i) )} \hat{\mathcal{l}}(X_i, D = 0)")

where
![\hat{e}(\cdot)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Chat%7Be%7D%28%5Ccdot%29 "\hat{e}(\cdot)")
is the propensity score and
![\hat{\mathcal{l}}(\cdot)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Chat%7B%5Cmathcal%7Bl%7D%7D%28%5Ccdot%29 "\hat{\mathcal{l}}(\cdot)")
is an outcome model for
![Y(1) - Y(0)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;Y%281%29%20-%20Y%280%29 "Y(1) - Y(0)")
regressed on covariates in the untreated subsample. Both these functions
are estimated using LASSO(`glmnet`).

## DGP

``` r
# %% # DGP
simstudy = \(N=200, p=100, s=5, theta=3,
      ρ = 0 # no parallel trends violation by default
    ){
    # sparse signal in pscore
    γ=c(s:1,rep(0,(p-s)))/s
    X = array(rnorm(N*p), dim=c(N,p))
    e = expit(X %*% γ)
    D  = rbinom(N,1,e)
    beta1 = γ+0.5
    ε1 = rnorm(N,0,0.1); ε2 = ε3 = ε1
    # potential outcomes
    Y11=Y01=Y00=rep(NA, N)
    Y00 = X %*% beta1 + ε1
    Y01 = Y00 + 1 + ρ * e + ε2      # pscore enters outcome model
    Y11 = theta + Y01 +ε3
    # outcomes at two time periods
    Y0 = Y00
    Y1 = Y01*(1-D) + Y11*D
    data.frame(Y1, Y0, D, X)
}
```

``` r
B = 100

# naive difference
tauhats00 = mcReplicate(B, {
    df = simstudy()
    with(df, mean(Y1[D==1]) - mean(Y1[D==0]) )
  }, mc.cores = 8)
```

``` r
tauhats0 = mcReplicate(B, {
    df = simstudy()
    # DiD
    with(df, (mean(Y1[D==1]) - mean(Y0[D==1])) -
             (mean(Y1[D==0]) - mean(Y0[D==0])) )
  }, mc.cores = 8)
```

``` r
tauhats1 = mcReplicate(B, {
    df = simstudy()
    ipwDID(df[, -(1:3)] %>% as.matrix(), df$D, df$Y1, df$Y0, xfit = F)
  }, mc.cores = 8)
```

``` r
tauhats2 = mcReplicate(B, {
    df = simstudy()
    ipwDID(df[, -(1:3)] %>% as.matrix(), df$D, df$Y1, df$Y0, xfit = T)
  }, mc.cores = 8)
```

``` r
tauhats3 = mcReplicate(B, {
    df = simstudy()
    aipwDID(df[, -(1:3)] %>% as.matrix(), df$D, df$Y1, df$Y0, xfit = F)
  }, mc.cores = 8)
```

``` r
tauhats4 = mcReplicate(B, {
    df = simstudy()
    aipwDID(df[, -(1:3)] %>% as.matrix(), df$D, df$Y1, df$Y0, xfit = T)
  }, mc.cores = 8)
```

``` r
θ = 3
ests = c(tauhats00, tauhats0, tauhats1, tauhats2, tauhats3, tauhats4)
bounds = quantile(ests, c(0.05, 0.95))
lb = 2.5; ub = 3.5

par(mfrow = c(3, 2))
hist(tauhats00, breaks=100, main="Naive Post", xlab="", ylab="", xlim = c(lb, ub))
abline(v = mean(tauhats00), col = 'blue'); abline(v = θ, col = 'red')
hist(tauhats0, breaks=100, main="DiD",     xlab="", ylab="", xlim = c(lb, ub))
abline(v = mean(tauhats0), col = 'blue'); abline(v = θ, col = 'red')

hist(tauhats1, breaks=100, main="FS: IPW", xlab="", ylab="", xlim = c(lb, ub))
abline(v = mean(tauhats1), col = 'blue'); abline(v = θ, col = 'red')
hist(tauhats2, breaks=100, main="XF: IPW",     xlab="", ylab="", xlim = c(lb, ub))
abline(v = mean(tauhats2), col = 'blue'); abline(v = θ, col = 'red')

hist(tauhats3, breaks=100, main="FS: AIPW",    xlab="", ylab="", xlim = c(lb, ub))
abline(v = mean(tauhats3), col = 'blue'); abline(v = θ, col = 'red')
hist(tauhats4, breaks=100, main="XF: AIPW",    xlab="", ylab="", xlim = c(lb, ub))
abline(v = mean(tauhats4), col = 'blue'); abline(v = θ, col = 'red')
```

![](readme_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

### Parallel Trends Violations

``` r
B = 100
# naive difference
tauhats00 = mcReplicate(B, {
    df = simstudy(ρ = 0.5)
    with(df, mean(Y1[D==1]) - mean(Y1[D==0]) )
  }, mc.cores = 8)
```

``` r
tauhats0 = mcReplicate(B, {
    df = simstudy(ρ = 0.5)
    # DiD
    with(df, (mean(Y1[D==1]) - mean(Y0[D==1])) -
             (mean(Y1[D==0]) - mean(Y0[D==0])) )
  }, mc.cores = 8)
```

``` r
tauhats1 = mcReplicate(B, {
    df = simstudy(ρ = 0.5)
    ipwDID(df[, -(1:3)] %>% as.matrix(), df$D, df$Y1, df$Y0, xfit = F)
  }, mc.cores = 8)
```

``` r
tauhats2 = mcReplicate(B, {
    df = simstudy(ρ = 0.5)
    ipwDID(df[, -(1:3)] %>% as.matrix(), df$D, df$Y1, df$Y0, xfit = T)
  }, mc.cores = 8)
```

``` r
tauhats3 = mcReplicate(B, {
    df = simstudy(ρ = 0.5)
    aipwDID(df[, -(1:3)] %>% as.matrix(), df$D, df$Y1, df$Y0, xfit = F)
  }, mc.cores = 8)
```

``` r
tauhats4 = mcReplicate(B, {
    df = simstudy(ρ = 0.5)
    aipwDID(df[, -(1:3)] %>% as.matrix(), df$D, df$Y1, df$Y0, xfit = T)
  }, mc.cores = 8)
```

``` r
θ = 3
ests = c(tauhats00, tauhats0, tauhats1, tauhats2, tauhats3, tauhats4)
bounds = quantile(ests, c(0.05, 0.95))
lb = 2.5; ub = 3.5

par(mfrow = c(3, 2))
hist(tauhats00, breaks=100, main="Naive Post", xlab="", ylab="", xlim = c(lb, ub))
abline(v = mean(tauhats00), col = 'blue'); abline(v = θ, col = 'red')
hist(tauhats0, breaks=100, main="DiD",     xlab="", ylab="", xlim = c(lb, ub))
abline(v = mean(tauhats0), col = 'blue'); abline(v = θ, col = 'red')

hist(tauhats1, breaks=100, main="FS: IPW", xlab="", ylab="", xlim = c(lb, ub))
abline(v = mean(tauhats1), col = 'blue'); abline(v = θ, col = 'red')
hist(tauhats2, breaks=100, main="XF: IPW",     xlab="", ylab="", xlim = c(lb, ub))
abline(v = mean(tauhats2), col = 'blue'); abline(v = θ, col = 'red')

hist(tauhats3, breaks=100, main="FS: AIPW",    xlab="", ylab="", xlim = c(lb, ub))
abline(v = mean(tauhats3), col = 'blue'); abline(v = θ, col = 'red')
hist(tauhats4, breaks=100, main="XF: AIPW",    xlab="", ylab="", xlim = c(lb, ub))
abline(v = mean(tauhats4), col = 'blue'); abline(v = θ, col = 'red')
```

![](readme_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

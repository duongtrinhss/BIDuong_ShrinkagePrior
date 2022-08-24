# BAYESIAN REGRESSION ====

## R function ====
library("matlab")
library("MCMCpack")
BayesRegrR <- function(y, X, nsave = 10000, nburn = 1000) {
  k <- ncol(X)
  n <- nrow(X)
  
  # =====| Initialize parameters
  B0 <- 1000*eye(k)
  b0 = zeros(k,1)
  alpha0 = 0.002
  delta0 = 0.002
  
  sigmasq = 1
  
  # =====| Storage space for Gibbs draws
  beta_draws = zeros(k,nsave)
  sigmasq_draws = zeros(1,nsave)
  
  #==========================================================================
  #====================| GIBBS ITERATIONS START HERE |=======================
  for (iter in 1:(nburn + nsave)) {
    #=====|Draw beta
    Bn <- solve(t(X)%*%X/sigmasq + solve(B0))
    bn <- Bn%*%(t(X)%*%y/sigmasq + solve(B0)%*%b0)
    H <- chol(Bn)
    beta <- bn + t(H)%*%matrix(rnorm(k),k,1)
    
    #=====|Draw sigma^2
    resid <- y - X%*%beta
    alphan <- alpha0 + n
    deltan <- delta0 + t(resid)%*%resid
    sigmasq <- rinvgamma(1, shape = alphan/2, scale = deltan/2)
    
    #=====|Save draws
    if (iter > nburn) {
      beta_draws[,iter-nburn] <- beta
      sigmasq_draws[,iter-nburn] <- sigmasq
    }
  }
  return(list(betadraws=beta_draws,sigmasqdraws=sigmasq_draws))
}
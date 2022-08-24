//[[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

#include "BayesRegr.h"

//[[Rcpp::export]]
Rcpp::List BayesRegr(
    const arma::colvec& y, // an lvalue reference to a colvec object (the ampersand & means "lvalue reference to")
    const arma::mat& X,
    int nsave,
    int nburn)
{
    //--------------------------------------------------------------
    //
    arma::uword n = X.n_rows, p = X.n_cols;
    // Initialize parameters
    arma::mat B0 = 1000*eye(p,p);
    arma::mat b0 = zeros(p,1);
    double alpha0 { 0.002 }, delta0 { 0.002 };
    arma::colvec Q = B0.diag();
    
    arma::colvec beta = zeros(p,1);
    double sigmasq = 1;
    
    // Process
    /*arma::mat Bn, bn, H, beta;*/
    arma::colvec resid;
    double alphan, deltan;
    
    // Return data structures
    arma::mat beta_draws = zeros(p,nsave);
    arma::mat sigmasq_draws = zeros(1,nsave);
    
    //-----------------GIBBS ITERATIONS START HERE--------------------
    for (arma::uword iter = 0; iter < (nburn + nsave); iter ++) {
        
        // Draw beta
        /*
        Bn = inv(X.t()*X/sigmasq + inv(B0));
        bn = Bn*X.t()*y/sigmasq + solve(B0,b0);
        H = chol(Bn);
        beta = bn + H.t()*randn(p,1);
        */
        beta = randn_gibbs(y, X, Q, sigmasq, n, p);
        
        // Draw sigmasq
        resid = y - X*beta;
        alphan = alpha0 + n;
        deltan = delta0 + arma::as_scalar(resid.t()*resid);
        sigmasq = 1/R::rgamma(alphan/2,2/deltan);
        
        // Save draws
        if (iter > nburn - 1) {
            beta_draws.col(iter-nburn) = beta;
            sigmasq_draws.col(iter-nburn) = sigmasq;
        }
    }
    return Rcpp::List::create(
        Rcpp::Named("betadraws") = beta_draws,
        Rcpp::Named("sigmasqdraws") = sigmasq_draws);
}


//[[Rcpp::export]]
arma::mat randn_gibbs(
    arma::colvec y,
    arma::mat X,
    arma::colvec lambda,
    double sigmasq,
    double n,
    double p)
{
    // Transformation
    y = y/sqrt(sigmasq);
    X = X/sqrt(sigmasq);
    
    // Sample Gausian posterior efficiently (Rue, 2001)
    arma::mat Q_star = X.t()*X;
    arma::mat Dinv = diagmat(1/lambda);
    arma::mat L = chol((Q_star + Dinv),"lower");
    arma::colvec v = solve(L,(y.t()*X).t());
    arma::colvec mu = solve(L.t(),v);
    arma::colvec u = solve(L.t(),randn<colvec>(p));
    arma::colvec Beta = mu + u;
    
    return std::move(Beta); //call 'std::move' explicitly to avoid copying local variable Beta?
}



//[[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

#include "BayesRegrSP.h"

arma::colvec randn_gibbs(
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
    
    return Beta; //call 'std::move' explicitly to avoid copying local variable Beta?
}


// [[Rcpp::export]]
Col<double> rrinvgauss(int n, double mu, double lambda){
    Col<double> random_vector(n);
    double z,y,x,u;
    for(int i=0; i<n; ++i){
        z=R::rnorm(0,1);
        y=z*z;
        x=mu+0.5*mu*mu*y/lambda - 0.5*(mu/lambda)*sqrt(4*mu*lambda*y+mu*mu*y*y);
        u=R::runif(0,1);
        if(u <= mu/(mu+x)){
            random_vector(i)=x;
        }else{
            random_vector(i)=mu*mu/x;
        };
    }
    return(random_vector);
}


Rcpp::List cts_prior(
    const arma::colvec beta,
    const double a,
    const double b,
    const double c,
    const double d,
    arma::colvec tausq,
    const int method)
{
    int p = beta.n_elem;
    /*
    arma::colvec Q(p,1);
    */
    arma::colvec Q = 0.01*ones(p,1);

    switch (method)
    {
        case 1:
            for (arma::uword j = 0; j < p; j++)
            {
                Q[j] = 1/R::rgamma(a + 1/2, 1/(b + pow(beta[j],2)/2));
            }
            break;
        case 2:
            double lsq = R::rgamma(c + p, d + 0.5 * sum(tausq));
            for (arma::uword j = 0; j < p; j++)
            {
                tausq[j] = 1/(rrinvgauss(1, sqrt(lsq/pow(beta[j],2)), lsq)[0]);
            }
            Q = tausq;
            break;
    }
    
    return Rcpp::List::create(
        Rcpp::Named("Q") = Q,
        Rcpp::Named("tausq") = tausq
    );
}

//[[Rcpp::export]]
Rcpp::List BayesRegrSP(
    const arma::colvec& y, // an lvalue reference to a colvec object (the ampersand & means "lvalue reference to")
    const arma::mat& X,
    int nsave,
    int nburn,
    std::string prior)
{
    //--------------------------------------------------------------
    //
    arma::uword n = X.n_rows, p = X.n_cols;
    // Define priors ----
    
    // arma::mat B0 = 0.01*eye(p,p);
    // arma::colvec Q = B0.diag(); // prior variances for beta
    arma::colvec Q = 0.01*ones(p,1); // prior variances for beta
    double as { 0.1 }, bs { 0.1 }; // prior for sigmasq
    double a {}, b {};
    double c {}, d {};
    arma::colvec tausq(p);
    int method {};
    // Prior types
    
    if (prior == "student-t")
    {
        prior = "st";
        a = 1;
        b = 0.0001;
        method = 1;
    }
    else if (prior == "lasso")
    {
        prior = "ls";
        c = 1;
        d = 1;
        tausq = 0.01*ones(p,1);
        method = 2;
    }
    /*
    else if (prior == "horseshoe-sl")
    {
        prior = "hss";
        
    }
    else if (prior == "horseshoe-mx")
    {
        prior = "hsm";
    }
    */
    else
    {
        stop("Unknown prior");
    }

    
    // Initialize parameters ----
    arma::colvec beta = zeros(p,1);
    double sigmasq = 1;
    
    // Process
    /*arma::mat Bn, bn, H, beta;*/
    arma::colvec resid;
    double as_n, bs_n;
    Rcpp::List L;
    
    // Return data structures
    arma::mat beta_draws = zeros(p,nsave);
    arma::mat sigmasq_draws = zeros(1,nsave);
    
    //-----------------GIBBS ITERATIONS START HERE--------------------
    for (arma::uword iter = 0; iter < (nburn + nsave); iter ++) {
        
        // Sample beta ----
        beta = randn_gibbs(y, X, Q, sigmasq, n, p);
        
        // Sample sigmasq ----
        resid = y - X*beta;
        as_n = as + n/2;
        bs_n = bs + arma::as_scalar(resid.t()*resid)/2;
        sigmasq = 1/R::rgamma(as_n,1/bs_n);
        
        // Sample variances ----
        L = cts_prior(beta,a,b,c,d,tausq,method);
        Q = Rcpp::as<colvec>(L["Q"]);
        tausq = Rcpp::as<colvec>(L["tausq"]);
        
        // Save draws
        if (iter > nburn - 1) {
            beta_draws.col(iter-nburn) = beta;
            sigmasq_draws.col(iter-nburn) = sigmasq;
        }
    }
    return Rcpp::List::create(
        Rcpp::Named("betadraws") = beta_draws,
        Rcpp::Named("sigmasqdraws") = sigmasq_draws,
        Rcpp::Named("Q") = Q,
        Rcpp::Named("prior") = prior,
        Rcpp::Named("a") = a,
        Rcpp::Named("b") = b,
        Rcpp::Named("c") = c,
        Rcpp::Named("d") = d);
}

 

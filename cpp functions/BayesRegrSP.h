#ifndef BayesRegrSP_h
#define BayesRegrSP_h

arma::colvec randn_gibbs(
    arma::colvec y,
    arma::mat X,
    arma::colvec lambda,
    double sigmasq,
    double n,
    double p);

Rcpp::List cts_prior(
    const arma::colvec beta,
    const double a,
    const double b,
    const double c,
    const double d,
    arma::colvec tausq,
    const int method);

Rcpp::List BayesRegrSP(
    const arma::colvec& y, // an lvalue reference to a colvec object (the ampersand & means "lvalue reference to")
    const arma::mat& X,
    int nsave,
    int nburn,
    std::string prior);





#endif /* BayesRegrSP_h */

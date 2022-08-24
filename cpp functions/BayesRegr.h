//
//  BayesRegr.hpp
//  
//
//  Created by Duong Trinh on 21/07/2022.
//

#ifndef BayesRegr_h
#define BayesRegr_h

Rcpp::List BayesRegr(const arma::colvec& y,
                     const arma::mat& X,
                     int nsave,
                     int nburn);

arma::mat randn_gibbs(
    arma::colvec y,
    arma::mat X,
    arma::colvec lambda,
    double sigmasq,
    double n,
    double p);

#endif /* BayesRegr_h */



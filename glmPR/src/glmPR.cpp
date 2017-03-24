#include <RcppArmadillo.h>
#include <cmath>
#include <omp.h>
#include "admm.h"
using namespace Rcpp;

// [[Rcpp::export]]
List glmPR(const arma::mat& X, const arma::colvec& y, double s = 0.5) {
    int M = X.n_rows, N = X.n_cols;
    ADMM admm(X, y);
    admm.train();
    arma::vec coef = admm.getZ();
    return List::create(Named("coefficients") = coef);
}

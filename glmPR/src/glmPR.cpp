#include <RcppArmadillo.h>
#include <cmath>
#include "admm.h"
using namespace Rcpp;

// [[Rcpp::export]]

List glmPR(const arma::mat& X, const arma::colvec& y, double lambda = 0.5, int threads = 4) {
	ADMM admm(X, y, lambda, threads);
	admm.train();
	arma::vec coef = admm.getZ();
	return List::create(Named("coefficients") = coef);
}
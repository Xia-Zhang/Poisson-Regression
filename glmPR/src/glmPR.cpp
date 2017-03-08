#include <RcppArmadillo.h>
#include "lbfgs.h"
#include <cmath>

using namespace Rcpp;

// Optimize the objective function of Poisson use lbfgs
static lbfgsfloatval_t evaluate(void *instance,
                                const lbfgsfloatval_t *beta,
                                lbfgsfloatval_t *g,
                                const int n,
                                const lbfgsfloatval_t step) {
    arma::mat YX = *(arma::mat*)instance;
    arma::mat Y = YX.col(0);
    arma::mat X = YX.cols(1, YX.n_cols - 1);
    int M = (signed)X.n_rows, N = (signed)X.n_cols;
    arma::vec innerMulti(M);

    lbfgsfloatval_t fbeta = 0.0;

    for (int i = 0; i < M; i++) {
        lbfgsfloatval_t tmp = 0;
        for (int j = 0; j < N; j++) {
            tmp += beta[j] * X(i, j);
        }
        innerMulti[i] = exp(tmp);
        fbeta += Y[i] * tmp - innerMulti[i];
    }
    fbeta = (-1) * fbeta / M;

    for (int j = 0; j < N; j++) {
        lbfgsfloatval_t tmp = 0.0;
        for (int i = 0; i < M; i++) {
            tmp += X(i, j) * ( Y[i] - innerMulti[i]);
        }
        g[j] = (-1) * tmp / M;
    }
    return fbeta;
}

arma::colvec getCoef(const arma::mat& X, const arma::colvec& y, double s) {
    int N = X.n_cols;
    lbfgsfloatval_t fbeta;
    lbfgsfloatval_t *beta = lbfgs_malloc(N);
    lbfgs_parameter_t param;
    arma::colvec coef(N);
    arma::mat Z = arma::join_rows(y, X);

    for (int i = 0; i < N; i++) {
        beta[i] = (i % 2) ? 1.0 : -1.0;
    }
    if (beta == NULL) {
        Rcpp::Rcout << "ERROR: Failed to allocate a memory block for variables.\n" << std::endl;
        return coef;
    }

    lbfgs_parameter_init(&param);
    param.orthantwise_c = s;
    param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;

    int retNum = lbfgs(N, beta, &fbeta, evaluate, NULL, & Z, &param);
    if (retNum != 0) {
        Rcpp::Rcout << "return Num: " << retNum << std::endl;
        return coef;
    }
    for (int i = 0; i < N; i++) {
        coef[i] = beta[i];
    }
    return coef;
}

// [[Rcpp::export]]
List glmPR(const arma::mat& X, const arma::colvec& y, double s = 0.0) {
    int M = X.n_rows, N = X.n_cols;

    arma::colvec coef = getCoef(X, y, s);
    arma::colvec res  = y - X*coef;

    double s2 = std::inner_product(res.begin(), res.end(), res.begin(), 0.0)/(M - N);

    arma::colvec std_err = arma::sqrt(s2 * arma::diagvec(arma::pinv(arma::trans(X)*X)));

    return List::create(Named("coefficients") = coef,
                        Named("stderr")       = std_err,
                        Named("df.residual")  = M - N);
}

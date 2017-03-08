\name{glmPR-Functions}
\alias{glmPR}
\title{glmPR}
\description{
  This funtion is use to do the Poisson Regression with L1 penalization.
  % These four functions are created when
  % \code{RcppArmadillo.package.skeleton()} is invoked to create a
  % skeleton packages.
}
\usage{
glmPR(X, ...)
# glmPR(formula = y ~ X)
# glmPR(X, y, s)
}
\arguments{
  \item{X}{the input matrix}
  \item{...}{
  glmPR(X, y, s) y: the response matrix;s: a constant scalar parameter to control the influence of L1-Norm
  glmPR(formula, s) formula: formula object
  }
  % \item{formula}{Or just use the formula to pass the relation and the data}
  % \item{s}{a constant scalar parameter to control the influence of L1-Norm}
}
\value{
  return the glmPR object, which can be use to predict.
}
\details{
  This function use the lbfgs algorithm, which is implemented in the C++ lib liblbfgs, to optimaze the objective function.
}
\references{
  liblbfgs, \url{http://www.chokkan.org/software/liblbfgs/} a library of Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS).

  Armadillo, \url{http://arma.sourceforge.net/}a templated C++ linear algebra library written by Conrad Sanderso.

  Eddelbuettel, Dirk (2013) Seamless R and C++ Integration with
    \pkg{Rcpp}. Springer, New York. ISBN 978-1-4614-6867-7.
}
\examples{
  x <- matrix( c(2, 4, 3, 5, 5, 7), nrow=3)
  y <- c(1.2, 1.8, 2.1)
  glmPR::glmPR(x, y)
  glmPR::glmPR(y ~ x)
  glmPR::glmPR(x, y, s = 1.0)
}
\author{Xia Zhang}
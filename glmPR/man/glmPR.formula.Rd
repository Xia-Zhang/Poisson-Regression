% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/glmPR.R
\name{glmPR.formula}
\alias{glmPR.formula}
\title{L1-penalized Poisson Regression input formula}
\usage{
\method{glmPR}{formula}(formula, data = list(), lambda = 1, threads = 4,
  ...)
}
\arguments{
\item{formula}{the formula object}

\item{data}{the data set}

\item{lambda}{a constant scalar parameter to control the influence of L1-Norm}

\item{threads}{the parallelize node number}

\item{...}{other parameter}
}
\value{
coefficients vector
}
\description{
L1-penalized Poisson Regression input formula
}
\examples{
x <- matrix(rnorm(100), ncol = 4)
y <- rpois(25, 3)
glmPR(y ~ x, 1.0, 4)

}

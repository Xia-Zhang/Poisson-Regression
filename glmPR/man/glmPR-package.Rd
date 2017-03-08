\name{glmPR-package}
\alias{glmPR-package}
\docType{package}
\title{glmPR}
\description{
L1-penalized Poisson Regression, completed by C++ and Rcpp. This is test solution of the 2017 GSOC project Graphical Models for Mixed Multi Modal Data \url{https://github.com/rstats-gsoc/gsoc2017/wiki/Graphical-Models-for-Mixed-Multi-Modal-Data}.
}
\details{
Use the RcppArmadillo and Rcpp to integrate C++ and R, and use the C++ lib liblbfgs to optimize the objective function of Poisson regression with L1 penalization.


The most important function of this package is glmPR. It accept the input matrix and the response vector as the arguments, like glmPR(x, y) or glmPR::glmPR(y ~ x). The return object contains the coefficient of model. You can also set the parameter "s" in function glmPR to blance the funtion F(x) and the L1-norm.

You can use the glmPR object to predict the new data also. Just use the function \code{predict(obj_glmPR, newx)}.
% \packageDESCRIPTION{glmPR}
% \packageIndices{glmPR}
% ~~ An overview of how to use the package, including the ~~
% ~~ most important functions ~~
}
\author{
Xia Zhang

Maintainer: zhangxia9403@gmail.com
}
\references{

liblbfgs, \url{http://www.chokkan.org/software/liblbfgs/} a library of Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS).

Armadillo, \url{http://arma.sourceforge.net/}a templated C++ linear algebra library written by Conrad Sanderso.

Eddelbuettel, Dirk (2013) Seamless R and C++ Integration with
  \pkg{Rcpp}. Springer, New York. ISBN 978-1-4614-6867-7.
}

\keyword{ package }
% \seealso{
% % ~~ Optional links to other man pages, e.g. ~~
% % ~~ \code{\link[<pkg>:<pkg>-package]{<pkg>}} ~~
% }
\examples{
# ~~ simple examples of the most important functions ~~
x <- matrix( c(2, 4, 3, 5, 5, 7), nrow=3)
y <- c(1.2, 1.8, 2.1)
glmPR::glmPR(x, y)
glmPR::glmPR(y ~ x)
fit <- glmPR::glmPR(x, y, s = 1.0)	
predict(fit, newx = x[1:2,])
}

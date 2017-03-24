glmPR
======

L1-penalized Poisson Regression, completed by C++ and Rcpp.


This is test solution of the 2017 GSOC project [Graphical Models for Mixed Multi Modal Data](https://github.com/rstats-gsoc/gsoc2017/wiki/Graphical-Models-for-Mixed-Multi-Modal-Data). 

Use the [RcppArmadillo](http://arma.sourceforge.net) and Rcpp to integrate C++ and R. And apply the alternating direction method of multipliers [(ADMM)](http://stanford.edu/~boyd/admm.html) to the algorithm, also use the bfgs algorithm to optimize the variables in ADMM. Use [testthat](https://github.com/hadley/testthat) to do the unit test of glmPR package.


You can use the Example below to test the function. Compared to glmnet and glm, 
seems that still have little difference because of the different optimize algorithm.

#Example

```{r}
library("glmPR")
library("glmnet")

x <- matrix(rnorm(500), ncol = 5)
y <- rpois(100, 3)
glmPR(x, y, 0.5, 1)
coef(glmnet(x, y, family = "poisson"), s = 0.5)

glmPR(x, y, 0, 4)
glm(y ~ x, family = "poisson")

```
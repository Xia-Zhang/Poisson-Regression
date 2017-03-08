glmPR
======

L1-penalized Poisson Regression, implemented by C++ and Rcpp. This is test solution of the 2017 GSOC project [Graphical Models for Mixed Multi Modal Data](https://github.com/rstats-gsoc/gsoc2017/wiki/Graphical-Models-for-Mixed-Multi-Modal-Data).  
Use the [RcppArmadillo](http://arma.sourceforge.net) and Rcpp to integrate C++ and R, and use the C++ lib [liblbfgs](http://www.chokkan.org/software/liblbfgs/) to optimize  the objective function of Poisson regression with L1 penalization.  
You can read the package help documentation to get more information. The testthat.R to test the function in glmPR package, and the test data PoissonExample.RData is from [glmnet](https://github.com/cran/glmnet/blob/master/data/PoissonExample.RData).
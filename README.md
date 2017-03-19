glmPR
======

L1-penalized Poisson Regression, completed by C++ and Rcpp.

This is test solution of the 2017 GSOC project [Graphical Models for Mixed Multi Modal Data](https://github.com/rstats-gsoc/gsoc2017/wiki/Graphical-Models-for-Mixed-Multi-Modal-Data). 

Use the [RcppArmadillo](http://arma.sourceforge.net) and Rcpp to integrate C++ and R. And apply the alternating direction method of multipliers [(ADMM)](http://stanford.edu/~boyd/admm.html) to the algorithm, but now can only run on a single machine. Also use the bfgs algorithm to optimize the variables in ADMM.  

You can read the package help documentation to get more information.
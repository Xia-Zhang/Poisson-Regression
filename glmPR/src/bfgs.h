#ifndef BFGS_H_
#define BFGS_H_

#include <RcppArmadillo.h>

class BFGS{
public:
	BFGS(double rho);
	arma::vec optimize(arma::mat originX, arma::vec originY, arma::vec Z, arma::vec U);

private:
	double rho;		//Lagrange ratio
	double epsilon;
	int maxLoop;
	arma::vec Z;
	arma::vec U;
	arma::mat originX;
	arma::vec originY;

	double f(arma::vec x);
	arma::vec g(arma::vec x);
	double armijo(arma::vec gk, arma::vec pk);
	double wolfe(arma::vec gk, arma::vec pk);
};

#endif
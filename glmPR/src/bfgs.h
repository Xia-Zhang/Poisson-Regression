#ifndef BFGS_H_
#define BFGS_H_

#include <RcppArmadillo.h>

class BFGS{
public:
	BFGS(double rho);
	~BFGS();
	arma::vec optimize(arma::vec originX, double originY, arma::vec Z, arma::vec U);

private:
	double rho;
	double epsilon;
	int maxLoop;
	arma::vec Z;
	arma::vec U;
	arma::vec originX;
	double originY;

	double f(arma::vec x);
	arma::vec g(arma::vec x);

};

#endif
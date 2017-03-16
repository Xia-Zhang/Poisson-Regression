#include <RcppArmadillo.h>
#include <cmath>
#include "admm.h"
#include "bfgs.h"

BFGS::BFGS(double rho) {
	// , double epsilon = 1e-5, in maxLoop = 1e5
	this->rho = rho;
}

arma::vec BFGS::optimize(arma::vec originX, double originY, arma::vec Z, arma::vec U) {
	this->originY = originY;
	this->originX = originX;
	this->U = U;
	this->Z = Z;
	int featuresNum = originX.n_elem;

	int iter = 0, m, mk;
	arma::mat Bk(featuresNum, featuresNum, arma::fill::eye);
	arma::vec x0, x, detak, yk, gk, pk;
	double beta = 0.55, sigma = 0.4;

	while (iter < maxLoop) {
		gk = g(x0);
		if (norm(gk) < epsilon) {
			break;
		}
		pk = -1.0 * arma::solve(Bk, gk);
		
		m = 0;
		mk = 0;
		while (m < 20) {
			if (f(x0 + pow(beta, m)*pk) <= f(x0) + sigma * pow(beta, m) * arma::dot(gk, pk) ) {
				mk = m;
				break;
			}
			m += 1;
		}
		x = x0 + pow(beta, mk) * pk;
		detak = x - x0;
		yk = g(x) - gk;

		if (arma::dot(detak, yk) > 0) {
			double ydeta = arma::dot(yk, detak);
			double detaBdeta = arma::dot(detak, (Bk * detak));

			Bk = Bk + (yk * yk.t()) / ydeta - (Bk * detak * detak.t() * Bk) / detak;
		}

		iter += 1;
		x0 = x;
	}
	return x0;
}

double BFGS::f(arma::vec x) {
	double tmp = arma::dot(originX, x);
	return exp(tmp) - originY * tmp + rho / 2 * arma::dot((x - Z + U), (x - Z + U));
}

arma::vec BFGS::g(arma::vec x) {
	double tmp = dot(originX, x);
	return originX * tmp - originY * originX + rho * (x + Z - U);
}
#include "admm.h"
#include "bfgs.h"

ADMM::ADMM(const arma::mat &data, const arma::vec &labels) {
	dataNum = data.n_rows;
	featuresNum = data.n_cols;

	this->data = data;
	this->labels = labels;

	x.set_size(dataNum, featuresNum);
	x.ones();
	z.set_size(featuresNum);
	z.ones();
	preZ.set_size(featuresNum);
	preZ.ones();
	u.set_size(dataNum, featuresNum);
	u.ones();

	lambda = 1.0;
	rho = 1.0;
	epsAbs = 1e-4;
	epsRel = 1e-2;
	maxLoop = 3;
}

void ADMM::setLambda(double l) {
	this->lambda = l;
}

void ADMM::setRho(double r) {
	this->rho = r;
}

void ADMM::setMaxloop(int m) {
	this->maxLoop = m;
}

void ADMM::train() {
	int iter = 0;
	while (iter < maxLoop) {
		Rcpp::Rcout << "#" << iter << std::endl;
		updateX();
		updateZ();
		updateU();
		if (stopCriteria() == true) {
			break;
		}
		iter++;
	}
}

void ADMM::updateX() {
	BFGS bfgs(rho);
	for (int i = 0; i < dataNum; i++) {
		arma::vec in = data.row(i).t();
		double out = labels(i);
		x.row(i) = (bfgs.optimize(in, out, z, u.row(i).t())).t();
	}
}

void ADMM::updateZ() {
	preZ = z;
	z = ((arma::sum(x, 0) + arma::sum(u, 0))/dataNum).t();
	softThreshold(lambda / (rho * dataNum), z);
	// Rcpp::Rcout << z;
}

void ADMM::updateU() {
	// TODO: change to distributed compute
	for (int i = 0; i < dataNum; i++) {
		u.row(i) = u.row(i) + x.row(i) - z.t();
	}
}

arma::vec ADMM::getZ() {
	return z;
}

void ADMM::softThreshold(double k, arma::vec &A) {
	for (int i = 0; i < A.n_elem; i++) {
		if (A[i] > k) {
			A[i] -= k;
		}
		else if (A[i] < (-1) * k) {
			A[i] += k;
		}
		else {
			A[i] = 0;
		}
	}
}
// Armadillo is a well written C++ library, where the matrix objects are properly destroyed

bool ADMM::stopCriteria() {
	double normX, normZ, normY, epsPri, epsDual, normR, normS;
	arma::vec s, r;
	normX = arma::norm(arma::sum(x, 0) / dataNum);
	normZ = arma::norm(z);
	normY = arma::norm(arma::sum(rho * u, 0) / dataNum);
	s = rho * (-1) * (z - preZ);
	r = (arma::sum(x, 0)).t() / dataNum + z;
	normS = arma::norm(s);
	normR = arma::norm(r);
	epsPri = sqrt(featuresNum) * epsAbs + epsRel * (normX > normZ ? normX : normZ);
	epsDual = sqrt(dataNum) * epsAbs + epsRel * normY;

	if (normR <= epsPri && normS <= epsDual) {
		return true;
	}
	return false;
}
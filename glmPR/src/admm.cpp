#include "admm.h"
#include "bfgs.h"

ADMM::ADMM(const arma::mat &data, const arma::vec labels) {
	// arma::vec labels, int maxLoop = 20, double lambda = 1.0, double rho = 1.0
	dataNum = data.n_cols;
	featuresNum = data.n_rows;

	this->data = data;
	this->labels = labels;
	x.set_size(dataNum, featuresNum);
	z.set_size(featuresNum);
	preZ.set_size(featuresNum);
	u.set_size(dataNum, featuresNum);

	this->maxLoop = maxLoop;
	this->lambda = lambda;
	this->rho = rho;
	epsAbs = 1e-4;
	epsRel = 1e-2;
}

ADMM::~ADMM() {
}

void ADMM::train() {
	int iter = 0;
	while (iter < maxLoop) {
		updateX();
		updateZ();
		updateU();
		if (stopCriteria() == true) {
			break;
		}
	}
}

void ADMM::updateX() {
	BFGS bfgs(rho);
	for (int i = 0; i < dataNum; i++) {
		arma::vec in = data.row(i);
		double out = labels(i);
		x.row(i) = (bfgs.optimize(in, out, z, u.row(i).t())).t();
	}
}

void ADMM::updateZ() {
	preZ = z;
	z = arma::sum(x, 0) + arma::sum(u, 0)/dataNum;
	softThreshold(lambda / (rho * dataNum), z);
}

void ADMM::updateU() {
	// TODO: change to distributed compute
	for (int i = 0; i < dataNum; i++) {
		u.row(i) = u.row(i) + x.row(i) - z;
	}
}

arma::vec ADMM::getZ() {
	return z;
}

void ADMM::softThreshold(double k, arma::vec &A) {
	for (int i = 0; i < A.n_elem; ++i) {
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
// rmadillo is a well written C++ library, where the matrix objects are properly destroyed

bool ADMM::stopCriteria() {
	double normX, normZ, normY, epsPri, epsDual, normR, normS;
	arma::vec s, r;
	normX = arma::norm(arma::sum(x, 0) / dataNum);
	normZ = arma::norm(z);
	normY = arma::norm(arma::sum(rho * u, 0) / dataNum);
	s = rho * (z - preZ);
	r = arma::sum(x, 0) / dataNum + z;
	normS = arma::norm(s);
	normR = arma::norm(r);
	epsPri = sqrt(featuresNum) * epsAbs + epsRel * (normX > normZ ? normX : normZ);
	epsDual = sqrt(dataNum) * epsAbs + epsRel * normY;

	if (normR <= epsPri && normS <= epsDual) {
		return true;
	}
	return false;
}
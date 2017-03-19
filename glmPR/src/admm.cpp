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

	lambda = 1;
	rho = 0.5;
	epsAbs = 1e-4;
	epsRel = 1e-2;
	maxLoop = 50;
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
	testt(data, labels, z);
	while (iter < maxLoop) {
		updateX();
		updateZ();
		updateU();
		if (stopCriteria() == true) {
			break;
		}
		testt(data, labels, z);
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
	// TODO reduceAll
	softThreshold(lambda / (rho * dataNum), z);
}

void ADMM::updateU() {
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

bool ADMM::stopCriteria() {
	double normX, normZ, normY, epsPri, epsDual, normR, normS;
	arma::vec s, r;
	// TODO reduceAll
	normX = calcuSqure(x);
	normZ = arma::norm(z);
	arma::mat y = u * rho;
	normY = calcuSqure(y);

	s = rho * (-1) * (z - preZ);
	normS = sqrt(dataNum) * arma::norm(s);
	normR = calcuSqureMinus(x, z);

	epsPri = sqrt(dataNum) * epsAbs + epsRel * (normX > sqrt(dataNum) * normZ ? normX : sqrt(dataNum) * normZ);
	epsDual = sqrt(dataNum) * epsAbs + epsRel * normY;
	if (normR <= epsPri && normS <= epsDual) {
		return true;
	}
	return false;
}

double ADMM::calcuSqure(arma::mat &A) {
	double ans = 0.0;
	for (int i = 0; i < A.n_rows; i++) {
		ans += arma::dot(A.row(i), A.row(i));
	}
	return sqrt(ans);
}

double ADMM::calcuSqureMinus(arma::mat &A, arma::vec &v) {
	double ans = 0.0;
	for (int i = 0; i < A.n_rows; i++) {
		ans += arma::dot((A.row(i).t() - v), (A.row(i).t() - v));
	}
	return sqrt(ans);
}
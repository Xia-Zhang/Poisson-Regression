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

	lambda = 0.2;
	rho = 1.0;
	epsAbs = 1e-4;
	epsRel = 1e-2;
	maxLoop = 20;
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
		// Rcpp::Rcout << "#" << iter << std::endl;
		updateX();
		// Rcpp::Rcout << "#" << x << std::endl;
		updateZ();
		// Rcpp::Rcout << "#" << z << std::endl;
		updateU();
		// Rcpp::Rcout << "#" << u << std::endl;
		if (stopCriteria() == true) {
			break;
		}
		iter++;
	}
}

void ADMM::updateX() {
	// Rcpp::Rcout << "Before X: " << x << std::endl;
	BFGS bfgs(rho);
	for (int i = 0; i < dataNum; i++) {
		arma::vec in = data.row(i).t();
		double out = labels(i);
		// Rcpp::Rcout << in << " " << out << " " << z << " " << u.row(i).t() << std::endl;
		x.row(i) = (bfgs.optimize(in, out, z, u.row(i).t())).t();
	}
	// Rcpp::Rcout << "After X: " << x << std::endl;
}

void ADMM::updateZ() {
	// Rcpp::Rcout << "Before Z: " << z << std::endl;
	preZ = z;
	z = ((arma::sum(x, 0) + arma::sum(u, 0))/dataNum).t();
	// Rcpp::Rcout << "Inner Z: " << z << std::endl;
	softThreshold(lambda / (rho * dataNum), z);
	// Rcpp::Rcout << "After Z: " << z << std::endl;
	Rcpp::Rcout << z << std::endl;
}

void ADMM::updateU() {
	// TODO: change to distributed compute
	// Rcpp::Rcout << "Before U: " << u << std::endl;
	for (int i = 0; i < dataNum; i++) {
		u.row(i) = u.row(i) + x.row(i) - z.t();
	}
	// Rcpp::Rcout << "After U: " << u << std::endl;
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
	r = (arma::sum(x, 0)).t() / dataNum - z;
	normS = arma::norm(s);
	normR = arma::norm(r);
	epsPri = sqrt(featuresNum) * epsAbs + epsRel * (normX > normZ ? normX : (normZ > 0) ? normZ : 0);
	epsDual = sqrt(dataNum) * epsAbs + epsRel * normY;
	if (normR <= epsPri && normS <= epsDual) {
		return true;
	}
	return false;
}
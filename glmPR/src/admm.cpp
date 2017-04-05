#define ARMA_DONT_PRINT_ERRORS
#include "admm.h"
#include "bfgs.h"
#include <omp.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

ADMM::ADMM(const arma::mat &data, const arma::vec &labels, double s, int threadNum) {
	dataNum = data.n_rows;
	featuresNum = data.n_cols;
	this->data = data;
	this->labels = labels;

	rho = 1;
	epsAbs = 1e-4;
	epsRel = 1e-2;
	maxLoop = 50;
	setThreadNumber(threadNum);
	setLambda(s);
	try{
		x.set_size(procN, featuresNum);
		x.ones();
		z.set_size(featuresNum);
		z.ones();
		preZ.set_size(featuresNum);
		preZ.ones();
		u.set_size(procN, featuresNum);
		u.ones();
		w.set_size(procN, featuresNum);
		w.zeros();
	} catch(...) {
		Rcpp::stop("memory allocation failed");
	}
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

void ADMM::setThreadNumber(int n) {
	if (n <= 0) {
		Rcpp::stop("the thread number shouldn't be 0 or less");
	}
	this->procN = n;
}

void ADMM::train() {
	int iter = 0;
	double t, tx, ty;

	omp_set_num_threads(procN);
	while (iter < maxLoop) {
		t = tx = ty = 0.0;
		#pragma omp parallel for reduction(+ : t, tx, ty)
			for (int node = 0; node < procN; node++) {
				updateU(node);
				updateX(node);
				w.row(node) = x.row(node) + u.row(node);
				t += arma::dot(x.row(node).t() - z, x.row(node).t() - z);
				tx += arma::dot(x.row(node).t(), x.row(node).t());
				ty += arma::dot(rho * u.row(node), rho * u.row(node));
			}
		updateZ();
		if (stopCriteria(t, tx, ty) == true) {
			break;
		}
		iter++;
	}
}

void ADMM::updateU(int node) {
	u.row(node) = u.row(node) + x.row(node) - z.t();
}

void ADMM::updateX(int node) {
	BFGS bfgs(rho);
	int start = node * (dataNum / procN);
	int end = (node == procN - 1) ? dataNum - 1 : (node + 1) * (dataNum / procN) - 1;
	if (start <= end) {
		arma::mat in = data.rows(start, end);
		arma::vec out = labels.subvec(start, end);
		x.row(node) = (bfgs.optimize(in, out, z, u.row(node).t())).t();
	}
}

void ADMM::updateZ() {
	preZ = z;
	z = (sum(w)).t() / procN;
	softThreshold(lambda / (rho * procN), z);
}

arma::vec ADMM::getZ() {
	return z;
}

void ADMM::softThreshold(double k, arma::vec &A) {
	for (unsigned int i = 0; i < A.n_elem; i++) {
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

bool ADMM::stopCriteria(double t, double tx, double ty) {
	double normX, normZ, normY, epsPri, epsDual, normR, normS;
	arma::vec s;

	normX = sqrt(tx);	
	normY = sqrt(ty);
	normZ = arma::norm(z);
	normR = sqrt(t);

	s = rho * (-1) * (z - preZ);
	normS = sqrt(procN) * arma::norm(s);

	epsPri = sqrt(dataNum) * epsAbs + epsRel * (normX > sqrt(procN) * normZ ? normX : sqrt(procN) * normZ);
	epsDual = sqrt(dataNum) * epsAbs + epsRel * normY;
	if (normR <= epsPri && normS <= epsDual) {
		return true;
	}
	return false;
}
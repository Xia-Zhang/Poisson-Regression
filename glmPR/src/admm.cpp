#include "admm.h"
#include "bfgs.h"
#include <omp.h>

ADMM::ADMM(const arma::mat &data, const arma::vec &labels) {
	dataNum = data.n_rows;
	featuresNum = data.n_cols;

	this->data = data;
	this->labels = labels;

	lambda = 0;
	rho = 0.5;
	epsAbs = 1e-4;
	epsRel = 1e-2;
	maxLoop = 50;
	procN = 4;

	x.set_size(procN, featuresNum);
	x.ones();
	z.set_size(featuresNum);
	z.ones();
	preZ.set_size(featuresNum);
	preZ.ones();
	u.set_size(procN, featuresNum);
	u.ones();
	w.set_size(featuresNum);
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
		Rcpp::stop("The thread number should more that 0.");
	}
	this->procN = n;
}

void testt(arma::mat data, arma::vec labels, arma::vec z) {
	double ans = 0.0;
	for (int i = 0; i < data.n_rows; i++) {
		ans += pow(exp(arma::dot(data.row(i), z)) - labels[i], 2);
	}
	//Rcpp::Rcout << "The MSE is " << ans << std::endl;
}

void ADMM::train() {
	int iter = 0;
	omp_set_num_threads(procN);
	testt(data, labels, z);
	while (iter < maxLoop) {
		w.zeros();
		t = tx = ty = 0.0;
		#pragma omp for reduction(+ : w, t, tx, ty)
			for (int node = 0; node < procN; node++) {
				updateU(node);
				updateX(node);
				w += x.row(node).t() + u.row(node).t();
				t += arma::dot(x.row(node).t() - z, x.row(node).t() - z);
				tx += arma::dot(x.row(node).t(), x.row(node).t());
				ty += arma::dot(rho * u.row(node), rho * u.row(node));
			}
		updateZ();
		if (stopCriteria() == true) {
			break;
		}
		testt(data, labels, z);
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
	z = w / procN;
	softThreshold(lambda / (rho * procN), z);
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
	// w, t, tx, ty 
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
	//Rcpp::Rcout << normR << " " << epsPri << " " << normS << " " << epsDual << std::endl;
	if (normR <= epsPri && normS <= epsDual) {
		return true;
	}
	return false;
}
#ifndef ADMM_H_
#define ADMM_H_

#include <RcppArmadillo.h>

class ADMM{
public:
	ADMM(const arma::mat &data, const arma::vec &labels);
	void train();
	void updateX();
	void updateZ();
	void updateU();
	arma::vec getZ();
	void softThreshold(double k, arma::vec &A);
	bool stopCriteria();

	void setLambda(double l);
	void setRho(double r);
	void setMaxloop(int m);

private:
	arma::mat x;	// dataNum * featuresNum
	arma::vec z;	// featuresNum
	arma::vec preZ;
	arma::mat u;	// dataNum * featuresNum
	double lambda;	// L1 trade off
	double rho;		// Lagrange ratio
	double epsAbs;
	double epsRel;

	int maxLoop;
	int featuresNum;
	int dataNum;
	arma::mat data;
	arma::vec labels;
};

#endif
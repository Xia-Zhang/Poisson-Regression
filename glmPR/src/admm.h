#ifndef ADMM_H_
#define ADMM_H_

#include <RcppArmadillo.h>

class ADMM{
public:
	ADMM(const arma::mat &data, const arma::vec labels);
	~ADMM();
	void train();
	void updateX();
	void updateZ();
	void updateU();
	arma::vec getZ();
	void softThreshold(double k, arma::vec &A);
	bool stopCriteria();

private:
	arma::mat x;	// dataNum * featuresNum
	arma::vec z;	// featuresNum
	arma::vec preZ;
	arma::mat u;	// dataNum * featuresNum
	double lambda;
	double rho;
	double epsAbs;
	double epsRel;

	int maxLoop;
	int featuresNum;
	int dataNum;
	arma::mat data;
	arma::vec labels;
};

#endif
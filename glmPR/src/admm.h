#ifndef ADMM_H_
#define ADMM_H_

#include <RcppArmadillo.h>

class ADMM{
public:
	ADMM(const arma::mat &data, const arma::vec &labels);
	void train();
	void updateU(int node);
	void updateX(int node);
	void updateZ();
	arma::vec getZ();

	void setLambda(double l);
	void setRho(double r);
	void setMaxloop(int m);

private:
	arma::mat x;	// procN * featuresNum
	arma::mat u;	// procN * featuresNum
	arma::vec z;	// featuresNum
	arma::vec preZ;
	arma::vec w;

	double t;
	double tx;
	double ty;
	double lambda;	// L1 trade off
	double rho;		// Lagrange ratio
	double epsAbs;
	double epsRel;

	int maxLoop;
	int featuresNum;
	int dataNum;
	int procN;
	arma::mat data;
	arma::vec labels;

	void softThreshold(double k, arma::vec &A);
	bool stopCriteria();
};

#endif
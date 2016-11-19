#include<iostream>
#include"HeatDiscrete.h"
#include<Eigen/Dense>
#include<functional>


using namespace std; 
using namespace Eigen; 


double gLeft(double x, double t)
{
	return exp(t - 2.0);
}
double gRight(double x, double t)
{
	return exp(t + 2.0);
}
double f(double x)
{
	return exp(x);
}

double fExact(double x, double tau)
{
	return exp(x + tau);
}

void max_error(double M, double N) {
	double xLeft = -2;
	double xRight = 2;
	double tauFinal = 1;
	double tauDividend = 0;


	HeatDiscrete x(xLeft, xRight, tauFinal, gLeft, gRight, f, M, N, fExact);
	cout.precision(10);
	//MatrixXd mat = x.ForwardEuler();
	//MatrixXd mat = x.BackwardEuler(SOR , 0.000001, 1.2);
	MatrixXd mat = x.CrankNicolson(SOR, 0.000001, 1.2);
	double max_error = 0;
	double RMS = 0;
	double tmp;
	double delta_x = (xRight - xLeft) / N;
	for (int k = 1; k <= N - 1; k++) {
		tmp = abs(mat(M, k) - exp(xLeft + k*delta_x + tauFinal));
		if (max_error < tmp) max_error = tmp;
	}
	double sum = 0;
	double exact;
	for (int k = 0; k <= N; k++) {
		exact = exp(xLeft + k*delta_x + tauFinal);
		sum += (mat(M, k) - exact)*(mat(M, k) - exact) / (exact*exact);
	}
	//cout << setprecision(9)<<mat << endl;
	cout << "\nmax_error:   " << cout.precision(10) << max_error << endl;
	cout << "\nRMS:    " << sqrt(1 / (N + 1)*sum) << endl;
}



int main() {
	// Q1
	double xLeft = -2;
	double xRight = 2;
	double tauFinal = 1;
	double tauDividend = 0;
	long M = 8;
	long N = 4;
	HeatDiscrete x(xLeft, xRight, tauFinal, gLeft, gRight, f, M, N, fExact);
	cout.precision(10);
	//MatrixXd a = x.ForwardEuler();
	//MatrixXd a1 = x.BackwardEuler(SOR , 0.000001, 1.2);
	MatrixXd a = x.CrankNicolson(LU , 0.000001, 1.2);
	cout << a << endl;


	// Q2
	max_error(8, 16);
	max_error(32, 32);
	max_error(128, 64);
	max_error(512, 128);

	return 0;
}

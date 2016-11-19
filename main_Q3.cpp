// This main cpp file is written for Q3 of HW8 (MTH9821 Fall2016)

#include<iostream>
#include"HeatDiscrete_Q3.hpp"
//#include"HeatDiscrete.h"
#include "BS.hpp"
#include<Eigen/Dense>
#include<functional>
#include <tuple>
#include <vector>


using namespace std;
using namespace Eigen;

// Set Global Variable
//const double S0 = 41, K = 40, T = 0.75, q = 0.02, vol = 0.35, r = 0.04;

double a = (r - q)*1.0 / (pow(vol, 2)) - 0.5;
double b = (r - q)*1.0 / (pow(vol, 2)) + 0.5 + 2 * 1.0*q / (pow(vol, 2));
double xLeft = log(S0*1.0 / K) + (vol - q - 0.5*vol*vol)*T - 3 * vol*sqrt(T);

double gLeft(double x, double t)
{
	return K*exp(a*xLeft + b*t)*(exp(-2 * r*t*1.0 / (pow(vol, 2))) - exp(xLeft - 2 * q*t*1.0 / pow(vol, 2)));
}
double gRight(double x, double t)
{
	return 0.0;
}
double f(double x)
{
	return  K*exp(a*x)*(max(1 - exp(x), 0.0));
}

int main() {
	
	double xRight = xLeft + 6 * vol*sqrt(T);
	double tauFinal = T*0.5*vol*vol;
	double tauDividend = 0;
	long M = 4;
	//long N = 8;
	double alpha_tmp = 0.45;
	HeatDiscrete x(xLeft, xRight, tauFinal, gLeft, gRight, f, M,alpha_tmp);
	cout.precision(10);
	MatrixXd A_FE = x.ForwardEuler();
	MatrixXd A_BELU = x.BackwardEuler(LU,0.000001,1.2);
	MatrixXd A_BESOR = x.BackwardEuler(SOR, 0.000001, 1.2);
	MatrixXd A_CNLU = x.CrankNicolson(LU, 0.000001, 1.2);
	MatrixXd A_CNSOR = x.CrankNicolson(SOR, 0.000001, 1.2);
	cout << "**************************\n";
	cout << "Forward Euler\n";
	cout << A_FE << endl;
	cout << "**************************\n";
	cout << "Backward Euler with LU\n";
	cout << A_BELU << endl;
	cout << "**************************\n";
	cout << "Backward Euler with SOR\n";
	cout << A_BESOR << endl;
	cout << "**************************\n";
	cout << "Crank Nicolson with LU\n";
	cout << A_CNLU << endl;
	cout << "**************************\n";
	cout << "Crank Nicolson with SOT\n";
	cout << A_CNSOR << endl;
	cout << "Done Part 1\n****************************\n";

	double BS_P = black_schole(0, T, S0, K, r, q, vol, "PUT", "V");

	for (int M = 4; M <= 256; M*=4) {
		HeatDiscrete x(xLeft, xRight, tauFinal, gLeft, gRight, f, M, alpha_tmp);
		cout.precision(10);
		MatrixXd A_FE = x.ForwardEuler();
		MatrixXd A_BELU = x.BackwardEuler(LU, 0.000001, 1.2);
		MatrixXd A_BESOR = x.BackwardEuler(SOR, 0.000001, 1.2);
		MatrixXd A_CNLU = x.CrankNicolson(LU, 0.000001, 1.2);
		MatrixXd A_CNSOR = x.CrankNicolson(SOR, 0.000001, 1.2);

		vector<double> error_FE, error_BELU, error_BESOR, error_CNLU, error_CNSOR;
		vector<double> greeks_FE, greeks_BELU, greeks_BESOR, greeks_CNLU, greeks_CNSOR;

		tuple<double, double, double> error;
		error= error_eu_pde(x, A_FE, BS_P);
		error_FE.push_back(get<0>(error));
		error_FE.push_back(get<1>(error));
		error_FE.push_back(get<2>(error));

		error = error_eu_pde(x,A_BELU, BS_P);
		error_BELU.push_back(get<0>(error));
		error_BELU.push_back(get<1>(error));
		error_BELU.push_back(get<2>(error));

		error = error_eu_pde(x, A_BESOR, BS_P);
		error_BESOR.push_back(get<0>(error));
		error_BESOR.push_back(get<1>(error));
		error_BESOR.push_back(get<2>(error));

		error = error_eu_pde(x, A_CNLU, BS_P);
		error_CNLU.push_back(get<0>(error));
		error_CNLU.push_back(get<1>(error));
		error_CNLU.push_back(get<2>(error));

		error = error_eu_pde(x, A_CNSOR, BS_P);
		error_CNSOR.push_back(get<0>(error));
		error_CNSOR.push_back(get<1>(error));
		error_CNSOR.push_back(get<2>(error));

		tuple<double, double, double> greeks;
		greeks = greeks_eu_pde(x, A_FE);
		greeks_FE.push_back(get<0>(greeks));
		greeks_FE.push_back(get<1>(greeks));
		greeks_FE.push_back(get<2>(greeks));

		greeks = greeks_eu_pde(x, A_BELU);
		greeks_BELU.push_back(get<0>(greeks));
		greeks_BELU.push_back(get<1>(greeks));
		greeks_BELU.push_back(get<2>(greeks));

		greeks = greeks_eu_pde(x, A_BESOR);
		greeks_BESOR.push_back(get<0>(greeks));
		greeks_BESOR.push_back(get<1>(greeks));
		greeks_BESOR.push_back(get<2>(greeks));

		greeks = greeks_eu_pde(x, A_CNLU);
		greeks_CNLU.push_back(get<0>(greeks));
		greeks_CNLU.push_back(get<1>(greeks));
		greeks_CNLU.push_back(get<2>(greeks));

		greeks = greeks_eu_pde(x, A_CNSOR);
		greeks_CNSOR.push_back(get<0>(greeks));
		greeks_CNSOR.push_back(get<1>(greeks));
		greeks_CNSOR.push_back(get<2>(greeks));

		cout << "************************\n";
		cout << "FE: \n";
		cout << "Errors of M = " << M << " are: \n";
		for (auto elem : error_FE) {
			cout << elem << ", ";
		}cout << endl;
		cout << "Greeks are: \n";
		for (auto elem : greeks_FE) {
			cout << elem << ", ";
		}cout << endl;

		cout << "************************\n";
		cout << "BELU: \n";
		cout << "Errors of M = " << M << " are: \n";
		for (auto elem : error_BELU) {
			cout << elem << ", ";
		}cout << endl;
		cout << "Greeks are: \n";
		for (auto elem : greeks_BELU) {
			cout << elem << ", ";
		}cout << endl;


		cout << "************************\n";
		cout << "BESOR: \n";
		cout << "Errors of M = " << M << " are: \n";
		for (auto elem : error_BESOR) {
			cout << elem << ", ";
		}cout << endl;
		cout << "Greeks are: \n";
		for (auto elem : greeks_BESOR) {
			cout << elem << ", ";
		}cout << endl;

		cout << "************************\n";
		cout << "CNLU: \n";
		cout << "Errors of M = " << M << " are: \n";
		for (auto elem : error_CNLU) {
			cout << elem << ", ";
		}cout << endl;
		cout << "Greeks are: \n";
		for (auto elem : greeks_CNLU) {
			cout << elem << ", ";
		}cout << endl;

		cout << "************************\n";
		cout << "CNSOR: \n";
		cout << "Errors of M = " << M << " are: \n";
		for (auto elem : error_CNSOR) {
			cout << elem << ", ";
		}cout << endl;
		cout << "Greeks are: \n";
		for (auto elem : greeks_CNSOR) {
			cout << elem << ", ";
		}cout << endl;



	}

	int stop;
	cout << "Enter number to stop (0): ";
	cin >> stop;
	//system("Pause");
}

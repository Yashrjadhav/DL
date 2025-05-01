//%%writefile Application.cpp
#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

// Simple linear regression (y = mx + c)
void linearRegression(const vector<double>& X, const vector<double>& Y, double& m, double& c) {
    int n = X.size();
    double sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;

    #pragma omp parallel for reduction(+:sumX, sumY, sumXY, sumXX)
    for (int i = 0; i < n; i++) {
        sumX += X[i];
        sumY += Y[i];
        sumXY += X[i] * Y[i];
        sumXX += X[i] * X[i];
    }

    // Calculate coefficients m and c using the normal equation
    m = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    c = (sumY - m * sumX) / n;
}

int main() {
    vector<double> X = {1, 2, 3, 4, 5};  // Input features
    vector<double> Y = {2, 4, 5, 4, 5};  // Target labels
    double m, c;

    linearRegression(X, Y, m, c);

    cout << "Linear Regression Model: y = " << m << "x + " << c << endl;
    return 0;
}

//!g++ -fopenmp Application.cpp -o Application
//!./Application

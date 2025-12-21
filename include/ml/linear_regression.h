#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "../core/matrix.h"

class LinearRegression {
public:
    Matrix theta; // Weights
    
    LinearRegression(size_t input_dim);

    // Analytical Solver: (X^T X)^-1 X^T y
    // Added lambda for L2 Regularization (Ridge)
    // If lambda = 0, it behaves like normal OLS
    void fit_analytical(const Matrix& X, const Matrix& y, float reg_lambda = 0.0f);

    // Iterative Solver: Gradient Descent
    // Added lambda for L2 Regularization (Ridge)
    void fit_sgd(const Matrix& X, const Matrix& y, size_t epochs, float lr, float reg_lambda = 0.0f);

    // Predictions
    Matrix predict(const Matrix& X) const;
};

#endif
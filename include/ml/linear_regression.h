#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "../core/matrix.h"

class LinearRegression {
public:
    Matrix theta; // Weights
    
    LinearRegression(size_t input_dim);

    // Analytical Solver: (X^T X)^-1 X^T y
    void fit_analytical(const Matrix& X, const Matrix& y);

    // Iterative Solver: Gradient Descent
    void fit_sgd(const Matrix& X, const Matrix& y, size_t epochs, float lr);

    // Predictions
    Matrix predict(const Matrix& X) const;
};

#endif
#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "../core/matrix.h"

class LogisticRegression {
public:
    Matrix theta;
    
    LogisticRegression(size_t input_dim);

    // Newton-Raphson Solver
    // Uses the Hessian Matrix (Second Derivative) to find the minimum
    // Update: theta = theta - H^-1 * gradient
    void fit_newton(const Matrix& X, const Matrix& y, size_t max_iters = 10);

    // Probability prediction (0.0 to 1.0)
    Matrix predict_proba(const Matrix& X) const;
    
    // Class prediction (0 or 1)
    Matrix predict(const Matrix& X, float threshold = 0.5f) const;
};

#endif
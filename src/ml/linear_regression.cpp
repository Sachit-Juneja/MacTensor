#include "../../include/ml/linear_regression.h"
#include <iostream>

LinearRegression::LinearRegression(size_t input_dim) 
    : theta(input_dim, 1) {
    // Initialize weights to small random values
    theta = Matrix::random(input_dim, 1);
    theta.scale(0.01f);
}

void LinearRegression::fit_analytical(const Matrix& X, const Matrix& y) {
    // Formula: theta = (X^T * X)^-1 * (X^T * y)
    // We solve the system: (X^T * X) * theta = (X^T * y)
    
    std::cout << "[LinearRegression] Fitting Analytical (Normal Equation)...\n";

    Matrix Xt = X.transpose();
    Matrix A = Xt.matmul(X); // The Hessian (X^T * X)
    Matrix b = Xt.matmul(y); // The Projections (X^T * y)

    // Solve A * theta = b
    // A is symmetric positive definite (usually), so we use Cholesky solver
    theta = A.solve_spd(b);
}

void LinearRegression::fit_sgd(const Matrix& X, const Matrix& y, size_t epochs, float lr) {
    // Gradient Descent: theta = theta - lr * gradient
    // Gradient: (1/m) * X^T * (X * theta - y)
    
    std::cout << "[LinearRegression] Fitting SGD (" << epochs << " epochs)...\n";
    
    size_t m = X.rows;
    float scaling_factor = 1.0f / (float)m;

    for (size_t i = 0; i < epochs; ++i) {
        // 1. Forward Pass
        Matrix predictions = X.matmul(theta);
        
        // 2. Compute Error
        Matrix error = predictions - y; // (pred - real)
        
        // 3. Compute Gradient
        // X^T * error
        Matrix Xt = X.transpose();
        Matrix gradient = Xt.matmul(error);
        
        // Average the gradient
        gradient.scale(scaling_factor);

        // 4. Update Weights
        // theta = theta - (gradient * lr)
        theta.subtract(gradient.scale(lr));

        // Optional: Print loss every 10%
        if (i % (epochs / 10) == 0) {
            float mse = error.dot(error) / (2.0f * m);
            std::cout << "Epoch " << i << " Loss: " << mse << "\n";
        }
    }
}

Matrix LinearRegression::predict(const Matrix& X) const {
    return X.matmul(theta);
}
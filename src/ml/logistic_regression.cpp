#include "../../include/ml/logistic_regression.h"
#include <cmath>
#include <iostream>

// Helper: Sigmoid Function
float sigmoid(float z) {
    return 1.0f / (1.0f + std::exp(-z));
}

LogisticRegression::LogisticRegression(size_t input_dim) 
    : theta(input_dim, 1) {
    theta = Matrix::random(input_dim, 1);
    theta.scale(0.01f);
}

Matrix LogisticRegression::predict_proba(const Matrix& X) const {
    Matrix z = X.matmul(theta);
    return z.apply(sigmoid);
}

Matrix LogisticRegression::predict(const Matrix& X, float threshold) const {
    Matrix probs = predict_proba(X);
    return probs.apply([threshold](float p) {
        return p >= threshold ? 1.0f : 0.0f;
    });
}

void LogisticRegression::fit_newton(const Matrix& X, const Matrix& y, size_t max_iters) {
    std::cout << "[LogisticRegression] Fitting Newton-Raphson...\n";
    
    size_t m = X.rows;
    // Newton's method converges quadratically, usually in very few steps (5-10)
    
    for (size_t i = 0; i < max_iters; ++i) {
        // 1. Hypothesis: h(x) = sigmoid(X * theta)
        Matrix h = predict_proba(X);
        
        // 2. Gradient: X^T * (h - y)
        // Note: No (1/m) scaling usually in Newton update as it cancels out with Hessian
        Matrix error = h - y;
        Matrix Xt = X.transpose();
        Matrix gradient = Xt.matmul(error);
        
        // 3. Hessian: H = X^T * S * X
        // S is diagonal matrix where S_ii = h_i * (1 - h_i)
        
        // Construct S (Diagonal Weight Matrix)
        Matrix S = Matrix::identity(m);
        for(size_t k=0; k<m; ++k) {
            float p = h(k, 0);
            S(k, k) = p * (1.0f - p); // derivative of sigmoid
        }
        
        // Compute H = X^T * (S * X)
        // This is the computational bottleneck (O(N^3) roughly)
        // But for "Classical ML" sizes it's fine.
        Matrix SX = S.matmul(X);
        Matrix H = Xt.matmul(SX);
        
        // Regularize Hessian to ensure invertibility (Levenberg-Marquardt trick)
        // H = H + lambda * I
        Matrix I = Matrix::identity(H.rows);
        Matrix H_reg = H + (I * 1e-4f); 

        // 4. Newton Update: theta = theta - H^-1 * gradient
        // Solve H * delta = gradient
        try {
            Matrix delta = H_reg.solve_spd(gradient);
            theta.subtract(delta);
        } catch (const std::exception& e) {
            std::cerr << "Newton Step Failed (Singular Hessian): " << e.what() << "\n";
            break;
        }

        // Print Log-Loss (Cost)
        // J = -1/m * sum(y*log(h) + (1-y)*log(1-h))
        // Simplified check: just norm of update
        if (i % 1 == 0) {
             // Quick convergence check
             std::cout << "Iter " << i << " | Gradient Norm: " << gradient.dot(gradient) << "\n";
        }
    }
}
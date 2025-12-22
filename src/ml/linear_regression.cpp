#include "../../include/ml/linear_regression.h"
#include <iostream>
#include <cmath>

LinearRegression::LinearRegression(size_t input_dim) 
    : theta(input_dim, 1) {
    // Initialize weights to small random values
    theta = Matrix::random(input_dim, 1);
    theta.scale(0.01f);
}

void LinearRegression::fit_analytical(const Matrix& X, const Matrix& y, float reg_lambda) {
    // Formula: theta = (X^T * X)^-1 * (X^T * y)
    // We solve the system: (X^T * X) * theta = (X^T * y)
    
    std::cout << "[LinearRegression] Fitting Analytical (Normal Equation)...\n";

    Matrix Xt = X.transpose();
    Matrix A = Xt.matmul(X); // The Hessian (X^T * X)
    Matrix b = Xt.matmul(y); // The Projections (X^T * y)

    // Ridge Regression Logic: A = A + lambda * I
    if (reg_lambda > 0.0f) {
        Matrix I = Matrix::identity(A.rows);
        Matrix Penalty = I * reg_lambda;
        A.add(Penalty); // Add penalty to the Hessian
    }

    // Solve A * theta = b
    // A is symmetric positive definite (usually), so we use Cholesky solver
    theta = A.solve_spd(b);
}

void LinearRegression::fit_sgd(const Matrix& X, const Matrix& y, size_t epochs, float lr, float reg_lambda) {
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

        if (reg_lambda > 0.0f) {
            // We usually don't regularize the bias term (intercept), 
            // but since our Matrix class doesn't track which row is bias, 
            // we apply it to all for now (standard for simple implementations).
            Matrix penalty = theta * reg_lambda;
            gradient.add(penalty);
        }

        // 4. Update Weights
        // theta = theta - (gradient * lr)
        theta.subtract(gradient * lr); // fixed syntax: gradient.scale(lr) modifies in place

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

// Helper: Soft Thresholding Operator
// S(z, gamma) = sign(z) * max(|z| - gamma, 0)
float soft_threshold(float z, float gamma) {
    if (z > 0 && z > gamma) return z - gamma;
    if (z < 0 && z < -gamma) return z + gamma;
    return 0.0f;
}

void LinearRegression::fit_lasso_cd(const Matrix& X, const Matrix& y, float reg_lambda, size_t epochs) {
    std::cout << "[LinearRegression] Fitting Lasso (Coordinate Descent, Lambda=" << reg_lambda << ")...\n";

    size_t m = X.rows;
    size_t n = X.cols;
    
    // Precompute sum of squares for each feature column (denominator term)
    // z_j = sum(x_ij^2)
    std::vector<float> z(n);
    for(size_t j=0; j<n; ++j) {
        Matrix xj = X.col(j);
        z[j] = xj.dot(xj); 
    }

    // Initialize theta (if not already)
    // Current Residual r = y - X * theta
    // We maintain the residual to avoid full re-computation every step (O(m) vs O(mn))
    Matrix predictions = X.matmul(theta);
    Matrix residual = y - predictions; // Deep copy for residual

    for(size_t epoch=0; epoch < epochs; ++epoch) {
        float max_change = 0.0f;

        for(size_t j=0; j<n; ++j) {
            float old_theta_j = theta(j, 0);
            
            // 1. Calculate 'rho': correlation between feature j and residual
            // We need the residual *without* the contribution of feature j.
            // r_without_j = r_current + X_j * theta_j
            // rho = X_j . r_without_j
            
            // Optimization: rho = (X_j . r_current) + (X_j . X_j) * theta_j
            // rho = (X_j . r_current) + z[j] * theta_j
            
            Matrix xj = X.col(j);
            float correlation = xj.dot(residual);
            float rho = correlation + z[j] * old_theta_j;
            
            // 2. Apply Soft Thresholding
            // The threshold depends on lambda and batch size m
            float gamma = reg_lambda * m; 
            float new_theta_j = soft_threshold(rho, gamma) / z[j]; // Normalize by variance
            
            // 3. Update Residual
            // r_new = r_old - X_j * (new_theta - old_theta)
            if (std::abs(new_theta_j - old_theta_j) > 1e-5) {
                float diff = new_theta_j - old_theta_j;
                // residual = residual - xj * diff
                // Use our vector ops
                // Note: We don't have a convenient 'add_scaled_vector' yet, so we use temp matrices
                // or just a loop for speed since it's a vector
                
                // Efficient loop update for residual
                for(size_t i=0; i<m; ++i) {
                    residual(i, 0) -= xj(i, 0) * diff;
                }
                
                theta(j, 0) = new_theta_j;
                max_change = std::max(max_change, std::abs(diff));
            }
        }

        // Check convergence
        if (max_change < 1e-4) {
            std::cout << "Converged at epoch " << epoch << "\n";
            break;
        }
    }
}
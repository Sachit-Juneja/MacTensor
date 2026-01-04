#include "../../include/ml/pca.h"
#include <iostream>
#include <cmath>

PCA::PCA(int n_components) 
    : n_components(n_components), components(1,1), mean(1,1), explained_variance(1,1) {}

void PCA::fit(const Matrix& X) {
    std::cout << "[PCA] Squashing " << X.cols << " dimensions down to " << n_components << "...\n";

    // 1. Calculate Mean (The 'Average Joe' of the dataset)
    // We need a row vector where each element is the mean of a column
    mean = Matrix(1, X.cols);
    for(size_t j=0; j<X.cols; ++j) {
        float sum = 0.0f;
        for(size_t i=0; i<X.rows; ++i) {
            sum += X(i, j);
        }
        mean(0, j) = sum / X.rows;
    }

    // 2. Center the data (X - Mean)
    // PCA essentially rotates data around the origin, so we gotta move the origin to the data center
    Matrix X_centered = X.clone(); // Deep copy
    for(size_t i=0; i<X.rows; ++i) {
        for(size_t j=0; j<X.cols; ++j) {
            X_centered(i, j) -= mean(0, j);
        }
    }

    // 3. The SVD Magic
    // We want X = U * S * Vt
    // The rows of Vt are our Principal Components (eigenvectors of covariance)
    // The values in S tell us how important each component is
    auto svd_res = X_centered.svd();

    // 4. Store the components
    // We only want the top 'n_components' rows from Vt
    // Vt is (features x features). We want (n_components x features)
    components = Matrix(n_components, X.cols);
    
    // Copy the top rows (most variance)
    for(size_t i=0; i<(size_t)n_components; ++i) {
        for(size_t j=0; j<X.cols; ++j) {
            components(i, j) = svd_res.Vt(i, j);
        }
    }

    // 5. Store Explained Variance (Singular values squared / (n-1))
    // S comes as a column vector (or matrix with 1 col)
    explained_variance = Matrix(n_components, 1);
    for(size_t i=0; i<(size_t)n_components; ++i) {
        float s = svd_res.S(i, 0);
        explained_variance(i, 0) = (s * s) / (X.rows - 1);
    }
    
    std::cout << "Explained Variance Ratio (Eigenvalues):\n";
    explained_variance.print();
}

Matrix PCA::transform(const Matrix& X) const {
    // Project data: X_new = (X - mean) * Components_Transposed
    
    // 1. Center the input
    Matrix X_centered = X.clone();
    for(size_t i=0; i<X.rows; ++i) {
        for(size_t j=0; j<X.cols; ++j) {
            X_centered(i, j) -= mean(0, j);
        }
    }

    // 2. Project
    // Components is (n_comp x features). We need (features x n_comp) for multiplication
    Matrix C_t = components.transpose();
    
    // Result is (samples x n_comp)
    return X_centered.matmul(C_t);
}

Matrix PCA::inverse_transform(const Matrix& X_transformed) const {
    // Reconstruct: X_original = X_transformed * Components + mean
    // This adds the "loss" back (well, it tries, but the noise is gone forever)
    
    Matrix X_recon = X_transformed.matmul(components); // (samples x features)
    
    // Add mean back
    for(size_t i=0; i<X_recon.rows; ++i) {
        for(size_t j=0; j<X_recon.cols; ++j) {
            X_recon(i, j) += mean(0, j);
        }
    }
    
    return X_recon;
}
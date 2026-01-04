#include "../../include/ml/gmm.h"
#include "../../include/ml/kmeans.h" // stealing kmeans for initialization
#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

GMM::GMM(int k, int max_iters) : k(k), max_iters(max_iters), weights(1, k) {}

float GMM::gaussian_pdf(const Matrix& x, const Matrix& mean, const Matrix& cov) const {
    // PDF = (1 / sqrt((2pi)^k * |Sigma|)) * exp(-0.5 * (x-u)^T * inv(Sigma) * (x-u))
    size_t d = x.cols;
    
    // determinant measures the "volume" of the covariance
    float det = cov.determinant();
    if (det <= 1e-9f) det = 1e-9f; // prevent explodey math
    
    Matrix inv_cov = cov.inverse(); 
    
    Matrix diff = x - mean; // distance from mean
    
    // Mahalanobis distance calculation
    // basically "how many standard deviations away is this point?"
    Matrix temp = diff.matmul(inv_cov);
    float mahalanobis = temp.dot(diff); // dot acts as vector mul for 1xD
    
    float coeff = 1.0f / std::sqrt(std::pow(2.0f * M_PI, (float)d) * det);
    return coeff * std::exp(-0.5f * mahalanobis);
}

void GMM::fit(const Matrix& X) {
    std::cout << "[GMM] Fitting " << k << " gaussians (EM Algorithm)...\n";
    
    // 1. Initialization
    // We use KMeans because random initialization for GMM is usually a disaster
    KMeans initializer(k, 10); 
    initializer.fit(X);
    
    means.clear();
    covariances.clear();
    
    for(int i=0; i<k; ++i) {
        means.push_back(initializer.centroids.row(i));
        
        // Start with spherical covariance (Identity)
        // This assumes features are uncorrelated initially
        covariances.push_back(Matrix::identity(X.cols));
        
        // Uniform weights
        weights(0, i) = 1.0f / k;
    }

    // 2. Expectation-Maximization Loop
    for(int iter=0; iter<max_iters; ++iter) {
        
        // --- E-Step: Guess which cluster each point belongs to ---
        Matrix responsibilities(X.rows, k);
        
        for(size_t i=0; i<X.rows; ++i) {
            float sum_prob = 0.0f;
            Matrix x = X.row(i);
            
            for(int j=0; j<k; ++j) {
                float pdf = gaussian_pdf(x, means[j], covariances[j]);
                float prob = weights(0, j) * pdf;
                responsibilities(i, j) = prob;
                sum_prob += prob;
            }
            
            // Normalize so rows sum to 1
            if (sum_prob > 1e-9f) {
                for(int j=0; j<k; ++j) responsibilities(i, j) /= sum_prob;
            } else {
                // Point is in the middle of nowhere, assign uniform
                for(int j=0; j<k; ++j) responsibilities(i, j) = 1.0f/k;
            }
        }

        // --- M-Step: Update Gaussian parameters to match the guess ---
        
        // Total weight assigned to each cluster
        std::vector<float> N_k(k, 0.0f);
        for(int j=0; j<k; ++j) {
            for(size_t i=0; i<X.rows; ++i) N_k[j] += responsibilities(i, j);
        }

        for(int j=0; j<k; ++j) {
            // Update Mean: Weighted average of points
            Matrix new_mean(1, X.cols); 
            for(size_t i=0; i<X.rows; ++i) {
                float r = responsibilities(i, j);
                // manual vector add scaling
                for(size_t d=0; d<X.cols; ++d) {
                    new_mean(0, d) += r * X(i, d);
                }
            }
            new_mean.scale(1.0f / N_k[j]);
            means[j] = new_mean;

            // Update Covariance: Weighted variance
            Matrix new_cov(X.cols, X.cols);
            for(size_t i=0; i<X.rows; ++i) {
                float r = responsibilities(i, j);
                Matrix diff = X.row(i) - new_mean;
                // Outer product: (x-u)^T * (x-u)
                Matrix term = diff.transpose().matmul(diff);
                term.scale(r);
                new_cov.add(term);
            }
            new_cov.scale(1.0f / N_k[j]);
            
            // Regularization: Add tiny value to diagonal so matrix doesn't collapse
            // (Singular covariance breaks the math)
            for(size_t d=0; d<X.cols; ++d) new_cov(d,d) += 1e-5f;
            
            covariances[j] = new_cov;

            // Update Weights
            weights(0, j) = N_k[j] / X.rows;
        }
    }
}

Matrix GMM::predict_proba(const Matrix& X) const {
    Matrix probs(X.rows, k);
    for(size_t i=0; i<X.rows; ++i) {
        float sum = 0.0f;
        Matrix x = X.row(i);
        for(int j=0; j<k; ++j) {
            float p = weights(0, j) * gaussian_pdf(x, means[j], covariances[j]);
            probs(i, j) = p;
            sum += p;
        }
        if (sum > 0) {
            for(int j=0; j<k; ++j) probs(i, j) /= sum;
        }
    }
    return probs;
}

std::vector<int> GMM::predict(const Matrix& X) const {
    Matrix probs = predict_proba(X);
    std::vector<int> preds(X.rows);
    
    for(size_t i=0; i<X.rows; ++i) {
        float max_p = -1.0f;
        int best_k = 0;
        for(int j=0; j<k; ++j) {
            if (probs(i, j) > max_p) {
                max_p = probs(i, j);
                best_k = j;
            }
        }
        preds[i] = best_k;
    }
    return preds;
}
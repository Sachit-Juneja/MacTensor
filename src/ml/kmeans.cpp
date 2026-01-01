#include "../../include/ml/kmeans.h"
#include <iostream>
#include <limits>
#include <cmath>
#include <random>

KMeans::KMeans(int k, int max_iters) 
    : k(k), max_iters(max_iters), centroids(k, 1) {} // Centroids size updated in fit()

float KMeans::dist_sq(const Matrix& v1, const Matrix& v2) const {
    // d^2 = ||a - b||^2 = (a-b) . (a-b)
    // We assume v1 and v2 are row vectors
    // To do this efficiently without allocating a new matrix for (v1-v2) every time:
    // We could calculate element-wise, but using our Matrix ops is cleaner for now.
    // Optimization: In a production kernel, we'd expand (a-b)^2 = a^2 + b^2 - 2ab
    
    // For simplicity and correctness with current API:
    Matrix diff = v1 - v2;
    return diff.dot(diff);
}

void KMeans::fit(const Matrix& X) {
    std::cout << "[KMeans] Fitting " << k << " clusters...\n";
    
    // 1. Initialize Centroids (Random Forgy Method)
    // Pick k unique random indices from X
    centroids = Matrix(k, X.cols);
    
    std::vector<int> indices(X.rows);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    for(int i=0; i<k; ++i) {
        Matrix random_point = X.row(indices[i]);
        // Copy data to centroid row
        for(size_t j=0; j<X.cols; ++j) {
            centroids(i, j) = random_point(0, j);
        }
    }

    // 2. Lloyd's Loop
    std::vector<int> assignments(X.rows);
    
    for(int iter=0; iter < max_iters; ++iter) {
        bool changed = false;
        
        // --- Step A: Assignment ---
        for(size_t i=0; i<X.rows; ++i) {
            Matrix point = X.row(i);
            
            float min_dist = std::numeric_limits<float>::max();
            int best_cluster = -1;

            for(int c=0; c<k; ++c) {
                float d = dist_sq(point, centroids.row(c));
                if (d < min_dist) {
                    min_dist = d;
                    best_cluster = c;
                }
            }

            if (assignments[i] != best_cluster) {
                assignments[i] = best_cluster;
                changed = true;
            }
        }

        // Check convergence (if no points moved clusters)
        if (!changed) {
            std::cout << "Converged at iteration " << iter << "\n";
            break;
        }

        // --- Step B: Update Centroids ---
        // Reset centroids to 0
        Matrix new_centroids(k, X.cols); // Starts with 0
        std::vector<int> counts(k, 0);

        for(size_t i=0; i<X.rows; ++i) {
            int cluster_id = assignments[i];
            Matrix point = X.row(i);
            
            // Accumulate sum
            // We iterate manually since 'add' works on full matrices
            for(size_t j=0; j<X.cols; ++j) {
                new_centroids(cluster_id, j) += point(0, j);
            }
            counts[cluster_id]++;
        }

        // Compute Mean
        for(int c=0; c<k; ++c) {
            if (counts[c] > 0) {
                float scale = 1.0f / counts[c];
                for(size_t j=0; j<X.cols; ++j) {
                    new_centroids(c, j) *= scale;
                }
            } else {
                // Handle empty cluster (re-initialize randomly or keep old)
                // Keeping old usually safer
                // or pick a random point again (not implemented here for brevity)
                std::cout << "Warning: Cluster " << c << " became empty.\n";
            }
        }
        centroids = new_centroids;
    }
}

std::vector<int> KMeans::predict(const Matrix& X) const {
    std::vector<int> preds(X.rows);
    for(size_t i=0; i<X.rows; ++i) {
        Matrix point = X.row(i);
        float min_dist = std::numeric_limits<float>::max();
        int best_cluster = -1;

        for(int c=0; c<k; ++c) {
            float d = dist_sq(point, centroids.row(c));
            if (d < min_dist) {
                min_dist = d;
                best_cluster = c;
            }
        }
        preds[i] = best_cluster;
    }
    return preds;
}
#include "../../include/ml/kmeans.h"
#include <iostream>
#include <limits>
#include <cmath>
#include <random>
#include <numeric> // for std::iota

KMeans::KMeans(int k, int max_iters) 
    : k(k), max_iters(max_iters), centroids(k, 1) {} 

float KMeans::dist_sq(const Matrix& v1, const Matrix& v2) const {
    // d^2 = ||a - b||^2
    // using our matrix subtraction and dot product
    Matrix diff = v1 - v2;
    return diff.dot(diff);
}

void KMeans::fit(const Matrix& X) {
    std::cout << "[KMeans] Fitting " << k << " clusters (K-Means++ Init)...\n";
    
    // allocate centroids
    centroids = Matrix(k, X.cols);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist_idx(0, (int)X.rows - 1);

    // --- Step 1: K-Means++ Initialization ---
    // 1a. Pick the first centroid completely at random
    int first_idx = dist_idx(gen);
    Matrix first_point = X.row(first_idx);
    
    // Manual copy because we don't have row-assignment operator yet
    for(size_t j=0; j<X.cols; ++j) {
        centroids(0, j) = first_point(0, j);
    }

    // 1b. Pick the remaining k-1 centroids
    for (int c = 1; c < k; ++c) {
        std::vector<float> min_sq_dists(X.rows);
        float sum_sq_dist = 0.0f;

        // For every point, find the squared distance to the *closest* existing centroid
        for (size_t i = 0; i < X.rows; ++i) {
            float min_d = std::numeric_limits<float>::max();
            Matrix point = X.row(i);
            
            // Check against all centroids we have chosen so far (0 to c-1)
            for (int j = 0; j < c; ++j) {
                float d = dist_sq(point, centroids.row(j));
                if (d < min_d) min_d = d;
            }
            min_sq_dists[i] = min_d;
            sum_sq_dist += min_d;
        }

        // Roulette Wheel Selection
        // Pick a random number between 0 and total distance. 
        // Walk through the array until we pass that number.
        std::uniform_real_distribution<> dist_prob(0.0f, sum_sq_dist);
        float target = dist_prob(gen);
        float cumulative = 0.0f;
        int chosen_idx = -1;

        for (size_t i = 0; i < X.rows; ++i) {
            cumulative += min_sq_dists[i];
            if (cumulative >= target) {
                chosen_idx = (int)i;
                break;
            }
        }
        
        // Safety net for floating point rounding errors
        if (chosen_idx == -1) chosen_idx = (int)X.rows - 1;

        // Assign the new centroid
        Matrix chosen_point = X.row(chosen_idx);
        for(size_t j=0; j<X.cols; ++j) {
            centroids(c, j) = chosen_point(0, j);
        }
    }

    // --- Step 2: Lloyd's Loop (Standard K-Means) ---
    std::vector<int> assignments(X.rows);
    
    for(int iter=0; iter < max_iters; ++iter) {
        bool changed = false;
        
        // A. Assignment Step
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

        if (!changed) {
            std::cout << "Converged at iteration " << iter << "\n";
            break;
        }

        // B. Update Step
        Matrix new_centroids(k, X.cols); // fills with 0
        std::vector<int> counts(k, 0);

        for(size_t i=0; i<X.rows; ++i) {
            int cluster_id = assignments[i];
            Matrix point = X.row(i);
            
            // Accumulate
            for(size_t j=0; j<X.cols; ++j) {
                new_centroids(cluster_id, j) += point(0, j);
            }
            counts[cluster_id]++;
        }

        // Average
        for(int c=0; c<k; ++c) {
            if (counts[c] > 0) {
                float scale = 1.0f / counts[c];
                for(size_t j=0; j<X.cols; ++j) {
                    new_centroids(c, j) *= scale;
                }
            } else {
                // If a cluster somehow ends up empty (orphan), keep the old centroid.
                // In production, we might re-randomize this, but let's trust the process.
                std::cout << "Warning: Cluster " << c << " is empty (forever alone).\n";
                // Copy old centroid back to avoid zeroing it out
                for(size_t j=0; j<X.cols; ++j) {
                    new_centroids(c, j) = centroids(c, j);
                }
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
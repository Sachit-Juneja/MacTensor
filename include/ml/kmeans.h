#ifndef KMEANS_H
#define KMEANS_H

#include "../core/matrix.h"
#include <vector>

class KMeans {
public:
    int k;              // Number of clusters
    int max_iters;
    Matrix centroids;   // The learned cluster centers (k x features)

    KMeans(int k = 3, int max_iters = 100);

    // Fits the model using Lloyd's Algorithm
    void fit(const Matrix& X);

    // Returns the closest cluster index for each sample
    std::vector<int> predict(const Matrix& X) const;

private:
    // Helper to calculate squared euclidean distance between two vectors
    float dist_sq(const Matrix& v1, const Matrix& v2) const;
};

#endif
#ifndef PCA_H
#define PCA_H

#include "../core/matrix.h"

class PCA {
public:
    int n_components;       // How many dimensions we want to keep (the target)
    Matrix components;      // The principal axes (eigenvectors), sorted by importance
    Matrix mean;            // The average of the training data (needed for centering)
    Matrix explained_variance; // How much info we kept (eigenvalues)

    PCA(int n_components = 2);

    // 1. Learn the projection matrix from X
    void fit(const Matrix& X);

    // 2. Project X into the lower dimensional space
    Matrix transform(const Matrix& X) const;

    // 3. Reconstruct original X from the compressed version (lossy!)
    Matrix inverse_transform(const Matrix& X_transformed) const;
};

#endif
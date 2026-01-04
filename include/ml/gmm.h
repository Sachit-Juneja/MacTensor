#ifndef GMM_H
#define GMM_H

#include "../core/matrix.h"
#include <vector>

class GMM {
public:
    int k;              // Number of gaussians (clusters)
    int max_iters;
    
    // GMM Parameters
    std::vector<Matrix> means;       // k means
    std::vector<Matrix> covariances; // k covariance matrices
    Matrix weights;                  // Mixing coeffs (how popular is each cluster?)

    GMM(int k = 3, int max_iters = 100);

    // Learn the gaussians using Expectation-Maximization (EM)
    void fit(const Matrix& X);
    
    // Get probabilities for each cluster
    Matrix predict_proba(const Matrix& X) const;
    
    // Get hard cluster assignments
    std::vector<int> predict(const Matrix& X) const;

private:
    // Probability Density Function of a Multivariate Gaussian
    float gaussian_pdf(const Matrix& x, const Matrix& mean, const Matrix& cov) const;
};

#endif
#ifndef GRADIENT_BOOSTING_H
#define GRADIENT_BOOSTING_H

#include "../core/matrix.h"
#include "decision_tree.h"
#include <vector>

class GradientBoostingRegressor {
public:
    int n_estimators;       // Number of trees
    float learning_rate;    // Eta (shrinkage)
    int max_depth;          // Depth of individual trees
    
    // The Ensemble
    std::vector<DecisionTreeRegressor> trees;
    float base_prediction; // F_0 (Initial prediction, usually mean)

    GradientBoostingRegressor(int n_estimators = 100, float learning_rate = 0.1f, int max_depth = 3);

    void fit(const Matrix& X, const Matrix& y);
    Matrix predict(const Matrix& X) const;
};

#endif
#include "../../include/ml/gradient_boosting.h"
#include <iostream>
#include <numeric>

GradientBoostingRegressor::GradientBoostingRegressor(int n_estimators, float learning_rate, int max_depth)
    : n_estimators(n_estimators), learning_rate(learning_rate), max_depth(max_depth), base_prediction(0.0f) {}

void GradientBoostingRegressor::fit(const Matrix& X, const Matrix& y) {
    std::cout << "[GradientBoosting] Fitting " << n_estimators << " trees (LR: " << learning_rate << ")...\n";
    
    // 1. Initialize F_0(x) with the mean of y
    float sum_y = 0.0f;
    for(size_t i=0; i<y.rows; ++i) sum_y += y(i, 0);
    base_prediction = sum_y / y.rows;

    // Current predictions (starts as constant mean)
    Matrix curr_preds(y.rows, 1);
    for(size_t i=0; i<y.rows; ++i) curr_preds(i, 0) = base_prediction;

    // 2. Iterate
    for(int t=0; t<n_estimators; ++t) {
        // Calculate Pseudo-Residuals: r_i = y_i - F_{t-1}(x_i)
        Matrix residuals = y - curr_preds;

        // Fit a weak learner (Decision Tree) to the RESIDUALS
        // Note: Using max_depth usually small (3-5) for boosting
        DecisionTreeRegressor tree(max_depth, 2); 
        tree.fit(X, residuals);

        // Update model: F_t(x) = F_{t-1}(x) + lr * tree.predict(x)
        Matrix tree_preds = tree.predict(X);
        
        // Apply learning rate
        tree_preds.scale(learning_rate);
        
        // Add to current predictions
        curr_preds.add(tree_preds); // curr_preds is modified in-place

        // Store the tree
        trees.push_back(tree);

        // Optional: Print loss
        if (t % 10 == 0) {
            // MSE Calculation
            Matrix diff = y - curr_preds;
            float mse = diff.dot(diff) / y.rows;
            std::cout << "Tree " << t << " | MSE: " << mse << "\n";
        }
    }
}

Matrix GradientBoostingRegressor::predict(const Matrix& X) const {
    // Start with base prediction
    Matrix preds(X.rows, 1);
    for(size_t i=0; i<X.rows; ++i) preds(i, 0) = base_prediction;

    // Add contributions from all trees
    for(const auto& tree : trees) {
        Matrix tree_preds = tree.predict(X);
        tree_preds.scale(learning_rate);
        preds.add(tree_preds);
    }

    return preds;
}
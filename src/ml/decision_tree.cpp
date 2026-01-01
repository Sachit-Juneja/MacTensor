#include "../../include/ml/decision_tree.h"
#include <limits>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <map>

DecisionTreeRegressor::DecisionTreeRegressor(int max_depth, int min_samples_split)
    : max_depth(max_depth), min_samples_split(min_samples_split) {}

void DecisionTreeRegressor::fit(const Matrix& X, const Matrix& y) {
    std::cout << "[DecisionTree] Building Tree (Max Depth: " << max_depth << ")...\n";
    root = build_tree(X, y, 0);
}

Matrix DecisionTreeRegressor::predict(const Matrix& X) const {
    Matrix preds(X.rows, 1);
    for(size_t i = 0; i < X.rows; ++i) {
        // Extract row as 1xCols view
        // Note: We need a way to access row data easily. 
        // We can just pass the matrix and index to helper.
        preds(i, 0) = predict_one(root, X.row(i));
    }
    return preds;
}

float DecisionTreeRegressor::predict_one(const std::shared_ptr<Node>& node, const Matrix& row) const {
    if (node->is_leaf) {
        return node->value;
    }
    
    // Check feature value
    float val = row(0, node->split_feature); // row is 1xN
    if (val <= node->split_threshold) {
        return predict_one(node->left, row);
    } else {
        return predict_one(node->right, row);
    }
}

float DecisionTreeRegressor::calculate_variance(const Matrix& y) const {
    if (y.rows == 0) return 0.0f;
    
    float mean = 0.0f;
    for(size_t i=0; i<y.rows; ++i) mean += y(i, 0);
    mean /= y.rows;

    float var = 0.0f;
    for(size_t i=0; i<y.rows; ++i) {
        float diff = y(i, 0) - mean;
        var += diff * diff;
    }
    return var / y.rows;
}

std::shared_ptr<Node> DecisionTreeRegressor::build_tree(const Matrix& X, const Matrix& y, int depth) {
    auto node = std::make_shared<Node>();
    
    // 1. Calculate Leaf Value (Mean of current y)
    float mean_y = 0.0f;
    if (y.rows > 0) {
        for(size_t i=0; i<y.rows; ++i) mean_y += y(i, 0);
        mean_y /= y.rows;
    }
    node->value = mean_y;

    // 2. Check Stopping Criteria
    if (depth >= max_depth || (int)X.rows < min_samples_split || X.rows <= 1) {
        node->is_leaf = true;
        return node;
    }

    // 3. Find Best Split
    float best_variance_red = -1.0f;
    size_t best_feature = 0;
    float best_threshold = 0.0f;
    
    // Current impurity (variance)
    float current_variance = calculate_variance(y);
    
    // Optimization: In a real library, we would sort columns to find thresholds faster.
    // For now, we iterate all features and samples (Greedy).
    for(size_t feat = 0; feat < X.cols; ++feat) {
        for(size_t row = 0; row < X.rows; ++row) {
            float thresh = X(row, feat);
            
            // Split Data
            // We need to count sizes first to allocate
            // Then fill. This is manual because we don't have boolean masking in Matrix yet.
            std::vector<size_t> left_idxs;
            std::vector<size_t> right_idxs;
            
            for(size_t i=0; i<X.rows; ++i) {
                if (X(i, feat) <= thresh) left_idxs.push_back(i);
                else right_idxs.push_back(i);
            }

            if (left_idxs.empty() || right_idxs.empty()) continue;

            // Create Y slices (we only need Y to calc variance)
            Matrix y_left(left_idxs.size(), 1);
            Matrix y_right(right_idxs.size(), 1);
            
            for(size_t k=0; k<left_idxs.size(); ++k) y_left(k,0) = y(left_idxs[k], 0);
            for(size_t k=0; k<right_idxs.size(); ++k) y_right(k,0) = y(right_idxs[k], 0);
            
            // Calculate Weighted Variance
            float var_left = calculate_variance(y_left);
            float var_right = calculate_variance(y_right);
            
            float weighted_var = (left_idxs.size() * var_left + right_idxs.size() * var_right) / X.rows;
            float reduction = current_variance - weighted_var;

            if (reduction > best_variance_red) {
                best_variance_red = reduction;
                best_feature = feat;
                best_threshold = thresh;
            }
        }
    }

    // If no gain, stop
    if (best_variance_red < 1e-5) {
        node->is_leaf = true;
        return node;
    }

    // 4. Perform the Split
    node->split_feature = best_feature;
    node->split_threshold = best_threshold;
    
    // Re-gather indices for the best split to build child matrices
    std::vector<size_t> left_idxs;
    std::vector<size_t> right_idxs;
    for(size_t i=0; i<X.rows; ++i) {
        if (X(i, best_feature) <= best_threshold) left_idxs.push_back(i);
        else right_idxs.push_back(i);
    }
    
    // Helper to slice matrix by rows
    auto slice_rows = [&](const Matrix& src, const std::vector<size_t>& idxs) {
        Matrix out(idxs.size(), src.cols);
        for(size_t i=0; i<idxs.size(); ++i) {
            for(size_t j=0; j<src.cols; ++j) {
                out(i, j) = src(idxs[i], j);
            }
        }
        return out;
    };

    Matrix X_left = slice_rows(X, left_idxs);
    Matrix y_left = slice_rows(y, left_idxs);
    Matrix X_right = slice_rows(X, right_idxs);
    Matrix y_right = slice_rows(y, right_idxs);

    // 5. Recurse
    node->left = build_tree(X_left, y_left, depth + 1);
    node->right = build_tree(X_right, y_right, depth + 1);

    return node;
}

DecisionTreeClassifier::DecisionTreeClassifier(int max_depth, int min_samples_split)
    : max_depth(max_depth), min_samples_split(min_samples_split) {}

void DecisionTreeClassifier::fit(const Matrix& X, const Matrix& y) {
    std::cout << "[DecisionTree] Building Classifier (Max Depth: " << max_depth << ")...\n";
    root = build_tree(X, y, 0);
}

Matrix DecisionTreeClassifier::predict(const Matrix& X) const {
    Matrix preds(X.rows, 1);
    for(size_t i = 0; i < X.rows; ++i) {
        preds(i, 0) = predict_one(root, X.row(i));
    }
    return preds;
}

float DecisionTreeClassifier::predict_one(const std::shared_ptr<Node>& node, const Matrix& row) const {
    if (node->is_leaf) return node->value;
    
    float val = row(0, node->split_feature);
    if (val <= node->split_threshold) {
        return predict_one(node->left, row);
    } else {
        return predict_one(node->right, row);
    }
}

// Calculate Gini Impurity = 1 - sum(probability_of_class_i ^ 2)
float DecisionTreeClassifier::calculate_gini(const Matrix& y) const {
    if (y.rows == 0) return 0.0f;

    std::map<float, int> counts;
    for(size_t i=0; i<y.rows; ++i) {
        counts[y(i,0)]++;
    }

    float impurity = 1.0f;
    for(auto const& [val, count] : counts) {
        float prob = (float)count / y.rows;
        impurity -= prob * prob;
    }
    return impurity;
}

float DecisionTreeClassifier::calculate_majority_class(const Matrix& y) const {
    if (y.rows == 0) return 0.0f;
    
    std::map<float, int> counts;
    for(size_t i=0; i<y.rows; ++i) {
        counts[y(i,0)]++;
    }

    float majority_class = counts.begin()->first;
    int max_count = -1;

    for(auto const& [val, count] : counts) {
        if (count > max_count) {
            max_count = count;
            majority_class = val;
        }
    }
    return majority_class;
}

std::shared_ptr<Node> DecisionTreeClassifier::build_tree(const Matrix& X, const Matrix& y, int depth) {
    auto node = std::make_shared<Node>();
    
    // 1. Calculate Leaf Value (Majority Class)
    node->value = calculate_majority_class(y);

    // 2. Check Stopping Criteria
    // Stop if max depth, not enough samples, or if pure (gini is 0)
    float current_gini = calculate_gini(y);
    if (depth >= max_depth || (int)X.rows < min_samples_split || current_gini < 1e-6) {
        node->is_leaf = true;
        return node;
    }

    // 3. Find Best Split (Minimize Weighted Gini)
    float best_gini_gain = -1.0f;
    size_t best_feature = 0;
    float best_threshold = 0.0f;
    
    // We want to maximize Gain = OldGini - WeightedNewGini
    // Which is equivalent to minimizing WeightedNewGini
    // But tracking Gain is often easier to reason about 0 cutoff
    
    for(size_t feat = 0; feat < X.cols; ++feat) {
        for(size_t row = 0; row < X.rows; ++row) {
            float thresh = X(row, feat);
            
            // Split indices
            std::vector<size_t> left_idxs;
            std::vector<size_t> right_idxs;
            
            for(size_t i=0; i<X.rows; ++i) {
                if (X(i, feat) <= thresh) left_idxs.push_back(i);
                else right_idxs.push_back(i);
            }

            if (left_idxs.empty() || right_idxs.empty()) continue;

            // Manual Slicing (Optimization: Do this without full copy in V2)
            Matrix y_left(left_idxs.size(), 1);
            Matrix y_right(right_idxs.size(), 1);
            for(size_t k=0; k<left_idxs.size(); ++k) y_left(k,0) = y(left_idxs[k], 0);
            for(size_t k=0; k<right_idxs.size(); ++k) y_right(k,0) = y(right_idxs[k], 0);

            // Calculate Weighted Gini
            float g_left = calculate_gini(y_left);
            float g_right = calculate_gini(y_right);
            
            float p_left = (float)left_idxs.size() / X.rows;
            float p_right = (float)right_idxs.size() / X.rows;
            
            float weighted_gini = p_left * g_left + p_right * g_right;
            float gain = current_gini - weighted_gini;

            if (gain > best_gini_gain) {
                best_gini_gain = gain;
                best_feature = feat;
                best_threshold = thresh;
            }
        }
    }

    if (best_gini_gain < 0) { // Why wuld this happen? No valid split found
        node->is_leaf = true;
        return node;
    }

    node->split_feature = best_feature;
    node->split_threshold = best_threshold;

    // Recurse
    std::vector<size_t> left_idxs, right_idxs;
    for(size_t i=0; i<X.rows; ++i) {
        if (X(i, best_feature) <= best_threshold) left_idxs.push_back(i);
        else right_idxs.push_back(i);
    }
    
    auto slice_rows = [&](const Matrix& src, const std::vector<size_t>& idxs) {
        Matrix out(idxs.size(), src.cols);
        for(size_t i=0; i<idxs.size(); ++i) 
            for(size_t j=0; j<src.cols; ++j) 
                out(i, j) = src(idxs[i], j);
        return out;
    };

    node->left = build_tree(slice_rows(X, left_idxs), slice_rows(y, left_idxs), depth + 1);
    node->right = build_tree(slice_rows(X, right_idxs), slice_rows(y, right_idxs), depth + 1);

    return node;
}
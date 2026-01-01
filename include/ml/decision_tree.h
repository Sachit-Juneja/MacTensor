#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include "../core/matrix.h"
#include <memory>

struct Node {
    bool is_leaf;
    float value;            // For leaf: the prediction (mean of y)
    size_t split_feature;   // Index of feature to split on
    float split_threshold;  // Value to split on
    std::shared_ptr<Node> left;
    std::shared_ptr<Node> right;

    Node() : is_leaf(false), value(0), split_feature(0), split_threshold(0) {}
};

class DecisionTreeRegressor {
public:
    std::shared_ptr<Node> root;
    int max_depth;
    int min_samples_split;

    DecisionTreeRegressor(int max_depth = 5, int min_samples_split = 2);

    void fit(const Matrix& X, const Matrix& y);
    Matrix predict(const Matrix& X) const;

private:
    // Recursive builder
    std::shared_ptr<Node> build_tree(const Matrix& X, const Matrix& y, int depth);
    
    // Helper to calculate variance of a target vector
    float calculate_variance(const Matrix& y) const;
    
    // Helper to traverse tree for a single sample
    float predict_one(const std::shared_ptr<Node>& node, const Matrix& row) const;
};

class DecisionTreeClassifier {
public:
    std::shared_ptr<Node> root;
    int max_depth;
    int min_samples_split;

    DecisionTreeClassifier(int max_depth = 5, int min_samples_split = 2);
    void fit(const Matrix& X, const Matrix& y);
    Matrix predict(const Matrix& X) const;

private:
    std::shared_ptr<Node> build_tree(const Matrix& X, const Matrix& y, int depth);
    
    // Gini Impurity: 1 - sum(p^2)
    float calculate_gini(const Matrix& y) const;
    
    // Returns the most frequent class in y
    float calculate_majority_class(const Matrix& y) const;
    
    float predict_one(const std::shared_ptr<Node>& node, const Matrix& row) const;
};

#endif
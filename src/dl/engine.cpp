#include "../../include/dl/engine.h"
#include <iostream>
#include <algorithm>

// Constructor
Value::Value(Matrix data, std::vector<ValuePtr> children, std::string op)
    : data(data), grad(data.rows, data.cols), children(children), op(op) {
    // Gradient starts at zero
    // We assume 'grad' has same shape as 'data'
    // (Broadcast logic would go here if we were fancy)
    
    // Initializer _backward to do nothing (for leaf nodes)
    _backward = [](){};
}

ValuePtr Value::create(Matrix d) {
    return std::make_shared<Value>(d);
}

// --- The Engine Logic (Topological Sort) ---

void build_topo(ValuePtr v, std::vector<ValuePtr>& topo, std::unordered_set<ValuePtr>& visited) {
    if (visited.find(v) != visited.end()) return;
    visited.insert(v);
    
    // Visit all children first (dependencies)
    for (auto child : v->children) {
        build_topo(child, topo, visited);
    }
    // Then add self
    topo.push_back(v);
}

void Value::backward() {
    // 1. Initialize gradient of the root node (usually Loss) to 1.0
    // dL/dL = 1
    // We assume this node is a scalar (1x1) for loss usually, but works generally too
    // Setting all grads to 1.0 for the start node
    for(size_t i=0; i<grad.rows; ++i)
        for(size_t j=0; j<grad.cols; ++j)
            grad(i,j) = 1.0f;

    // 2. Topological Sort
    // We need to order nodes so we compute dependencies before dependants
    std::vector<ValuePtr> topo;
    std::unordered_set<ValuePtr> visited;
    build_topo(shared_from_this(), topo, visited);
    
    // 3. Reverse pass
    // Go through the list in reverse order (End -> Start)
    std::reverse(topo.begin(), topo.end());
    
    for (auto v : topo) {
        v->_backward();
    }
}

// --- Operations ---

ValuePtr Value::add(ValuePtr other) {
    // Forward Pass: Z = A + B
    Matrix out_data = data + other->data;
    
    ValuePtr out = std::make_shared<Value>(out_data, std::vector<ValuePtr>{shared_from_this(), other}, "+");
    
    // Backward Pass Logic
    // dL/dA += dL/dZ * 1
    // dL/dB += dL/dZ * 1
    // We use += because a node might be used multiple times (gradients accumulate!)
    out->_backward = [this, other, out]() {
        this->grad.add(out->grad);
        other->grad.add(out->grad);
    };
    
    return out;
}

ValuePtr Value::matmul(ValuePtr other) {
    // Forward Pass: Z = A @ B
    Matrix out_data = data.matmul(other->data);
    
    ValuePtr out = std::make_shared<Value>(out_data, std::vector<ValuePtr>{shared_from_this(), other}, "matmul");
    
    // Backward Pass Logic (The Matrix Calculus part)
    // If Z = A @ B
    // dL/dA = dL/dZ @ B.T
    // dL/dB = A.T @ dL/dZ
    out->_backward = [this, other, out]() {
        Matrix Bt = other->data.transpose();
        Matrix At = this->data.transpose();
        
        Matrix grad_A = out->grad.matmul(Bt);
        Matrix grad_B = At.matmul(out->grad);
        
        this->grad.add(grad_A);
        other->grad.add(grad_B);
    };
    
    return out;
}

ValuePtr Value::relu() {
    // Forward Pass: Z = max(0, A)
    // We define ReLU logic inline here or assume Matrix has apply()
    Matrix out_data = data.apply([](float x){ return x > 0.0f ? x : 0.0f; });
    
    ValuePtr out = std::make_shared<Value>(out_data, std::vector<ValuePtr>{shared_from_this()}, "ReLU");
    
    // Backward Pass
    // dL/dA = dL/dZ * (1 if A > 0 else 0)
    out->_backward = [this, out]() {
        // We need the input mask (where was data > 0?)
        // Standard ReLU gradient calculation
        for(size_t i=0; i<data.rows; ++i) {
            for(size_t j=0; j<data.cols; ++j) {
                float val = data(i,j);
                float incoming_grad = out->grad(i,j);
                if (val > 0) {
                    this->grad(i,j) += incoming_grad;
                }
                // else 0, so add nothing
            }
        }
    };
    
    return out;
}

// Operators
ValuePtr operator+(ValuePtr a, ValuePtr b) {
    return a->add(b);
}

ValuePtr operator*(ValuePtr a, ValuePtr b) {
    return a->matmul(b);
}
#include "../../include/dl/engine.h"
#include <iostream>
#include <algorithm>

Value::Value(Matrix data, std::vector<ValuePtr> children, std::string op)
    : data(data), grad(data.rows, data.cols), children(children), op(op) {
    
    // Initialize gradient to 0
    // (We use a simple loop or fill since we don't have a fill() method yet)
    for(size_t i=0; i<grad.rows; ++i)
        for(size_t j=0; j<grad.cols; ++j)
            grad(i,j) = 0.0f;
            
    // Default backward does nothing (for leaf nodes)
    _backward = [](){};
}

ValuePtr Value::create(Matrix d) {
    return std::make_shared<Value>(d);
}

// Topological Sort helper
void build_topo(ValuePtr v, std::vector<ValuePtr>& topo, std::unordered_set<ValuePtr>& visited) {
    if (visited.find(v) != visited.end()) return;
    visited.insert(v);
    
    for (auto child : v->children) {
        build_topo(child, topo, visited);
    }
    topo.push_back(v);
}

void Value::backward() {
    // 1. Set gradient of the final node (Loss) to 1.0
    for(size_t i=0; i<grad.rows; ++i)
        for(size_t j=0; j<grad.cols; ++j)
            grad(i,j) = 1.0f;

    // 2. Build the graph order
    std::vector<ValuePtr> topo;
    std::unordered_set<ValuePtr> visited;
    build_topo(shared_from_this(), topo, visited);
    
    // 3. Run backward pass in reverse order
    std::reverse(topo.begin(), topo.end());
    for (auto v : topo) {
        v->_backward();
    }
}

// --- Operations ---

ValuePtr Value::add(ValuePtr other) {
    // Z = A + B
    Matrix out_data = data + other->data;
    ValuePtr out = std::make_shared<Value>(out_data, std::vector<ValuePtr>{shared_from_this(), other}, "+");
    
    // dL/dA += dL/dZ
    // dL/dB += dL/dZ
    out->_backward = [this, other, out]() {
        this->grad.add(out->grad);
        other->grad.add(out->grad);
    };
    return out;
}

ValuePtr Value::matmul(ValuePtr other) {
    // Z = A @ B
    Matrix out_data = data.matmul(other->data);
    ValuePtr out = std::make_shared<Value>(out_data, std::vector<ValuePtr>{shared_from_this(), other}, "matmul");
    
    // dL/dA = dL/dZ @ B.T
    // dL/dB = A.T @ dL/dZ
    out->_backward = [this, other, out]() {
        this->grad.add(out->grad.matmul(other->data.transpose()));
        other->grad.add(this->data.transpose().matmul(out->grad));
    };
    return out;
}

ValuePtr Value::relu() {
    // Z = ReLU(A)
    Matrix out_data = data.apply([](float x){ return x > 0.0f ? x : 0.0f; });
    ValuePtr out = std::make_shared<Value>(out_data, std::vector<ValuePtr>{shared_from_this()}, "ReLU");
    
    // dL/dA = dL/dZ * (1 if A > 0 else 0)
    out->_backward = [this, out]() {
        for(size_t i=0; i<data.rows; ++i) {
            for(size_t j=0; j<data.cols; ++j) {
                if(data(i,j) > 0) this->grad(i,j) += out->grad(i,j);
            }
        }
    };
    return out;
}

ValuePtr operator+(ValuePtr a, ValuePtr b) { return a->add(b); }
ValuePtr operator*(ValuePtr a, ValuePtr b) { return a->matmul(b); }
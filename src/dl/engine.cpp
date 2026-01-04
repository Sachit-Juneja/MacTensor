#include "../../include/dl/engine.h"
#include <iostream>
#include <algorithm>

Value::Value(Matrix data, std::vector<ValuePtr> children, std::string op)
    : data(data), grad(data.rows, data.cols), children(children), op(op) {
    
    // Init gradient to 0
    for(size_t i=0; i<grad.rows; ++i)
        for(size_t j=0; j<grad.cols; ++j)
            grad(i,j) = 0.0f;
            
    _backward = [](){};
}

ValuePtr Value::create(Matrix d) {
    return std::make_shared<Value>(d);
}

void build_topo(ValuePtr v, std::vector<ValuePtr>& topo, std::unordered_set<ValuePtr>& visited) {
    if (visited.find(v) != visited.end()) return;
    visited.insert(v);
    for (auto child : v->children) build_topo(child, topo, visited);
    topo.push_back(v);
}

void Value::backward() {
    // [FIXED] Only set gradient to 1.0 if it hasn't been touched (all zeros).
    // This allows manual gradient injection (like dL/dPred) to persist.
    bool is_zero = true;
    for(size_t i=0; i<grad.rows; ++i) {
        for(size_t j=0; j<grad.cols; ++j) {
            if (grad(i,j) != 0.0f) {
                is_zero = false;
                break;
            }
        }
    }

    if (is_zero) {
        for(size_t i=0; i<grad.rows; ++i)
            for(size_t j=0; j<grad.cols; ++j)
                grad(i,j) = 1.0f;
    }

    // 2. Topo Sort
    std::vector<ValuePtr> topo;
    std::unordered_set<ValuePtr> visited;
    build_topo(shared_from_this(), topo, visited);
    
    // 3. Backprop
    std::reverse(topo.begin(), topo.end());
    for (auto v : topo) {
        v->_backward();
    }
}

// --- Ops ---

ValuePtr Value::add(ValuePtr other) {
    // Deep copy is handled by Matrix operator+, so this is safe
    Matrix out_data = data + other->data; 
    ValuePtr out = std::make_shared<Value>(out_data, std::vector<ValuePtr>{shared_from_this(), other}, "+");
    
    out->_backward = [this, other, out]() {
        this->grad.add(out->grad);
        other->grad.add(out->grad);
    };
    return out;
}

ValuePtr Value::matmul(ValuePtr other) {
    Matrix out_data = data.matmul(other->data);
    ValuePtr out = std::make_shared<Value>(out_data, std::vector<ValuePtr>{shared_from_this(), other}, "matmul");
    
    out->_backward = [this, other, out]() {
        // dL/dA = dL/dZ * B^T
        this->grad.add(out->grad.matmul(other->data.transpose()));
        // dL/dB = A^T * dL/dZ
        other->grad.add(this->data.transpose().matmul(out->grad));
    };
    return out;
}

ValuePtr Value::relu() {
    Matrix out_data = data.apply([](float x){ return x > 0.0f ? x : 0.0f; });
    ValuePtr out = std::make_shared<Value>(out_data, std::vector<ValuePtr>{shared_from_this()}, "ReLU");
    
    out->_backward = [this, out]() {
        // dL/dA = dL/dZ * (1 if A>0 else 0)
        // We rely on the input data stored in 'this' (A)
        for(size_t i=0; i<data.rows; ++i) {
            for(size_t j=0; j<data.cols; ++j) {
                if(data(i,j) > 0) 
                    this->grad(i,j) += out->grad(i,j);
            }
        }
    };
    return out;
}

ValuePtr operator+(ValuePtr a, ValuePtr b) { return a->add(b); }
ValuePtr operator*(ValuePtr a, ValuePtr b) { return a->matmul(b); }
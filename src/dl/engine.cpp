#include "../../include/dl/engine.h"
#include <iostream>
#include <algorithm>
#include <numeric>

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

ValuePtr Value::sub(ValuePtr other) {
    Matrix out_data = data - other->data;
    ValuePtr out = std::make_shared<Value>(out_data, std::vector<ValuePtr>{shared_from_this(), other}, "-");
    
    out->_backward = [this, other, out]() {
        this->grad.add(out->grad);
        // d(A-B)/dB = -1
        // We don't have a 'sub' or 'neg' for matrix in place yet, so scale -1 and add
        Matrix neg_grad = out->grad * -1.0f; 
        other->grad.add(neg_grad);
    };
    return out;
}

ValuePtr Value::mul(ValuePtr other) {
    // Element-wise multiplication (Hadamard)
    Matrix out_data = data.hadamard(other->data);
    ValuePtr out = std::make_shared<Value>(out_data, std::vector<ValuePtr>{shared_from_this(), other}, "*");
    
    out->_backward = [this, other, out]() {
        // d(A*B)/dA = B * dOut
        this->grad.add(other->data.hadamard(out->grad));
        other->grad.add(this->data.hadamard(out->grad));
    };
    return out;
}

ValuePtr Value::pow(float exponent) {
    // x^n
    Matrix out_data = data.apply([exponent](float x){ return std::pow(x, exponent); });
    ValuePtr out = std::make_shared<Value>(out_data, std::vector<ValuePtr>{shared_from_this()}, "^" + std::to_string(exponent));
    
    out->_backward = [this, out, exponent]() {
        // d(x^n)/dx = n * x^(n-1) * grad
        Matrix local_grad = data.apply([exponent](float x){ 
            return exponent * std::pow(x, exponent - 1.0f); 
        });
        this->grad.add(local_grad.hadamard(out->grad));
    };
    return out;
}

ValuePtr Value::exp() {
    // e^x
    Matrix out_data = data.apply([](float x){ return std::exp(x); });
    ValuePtr out = std::make_shared<Value>(out_data, std::vector<ValuePtr>{shared_from_this()}, "exp");
    
    out->_backward = [this, out]() {
        // d(e^x)/dx = e^x = out
        this->grad.add(out->data.hadamard(out->grad));
    };
    return out;
}

ValuePtr Value::log() {
    // ln(x)
    Matrix out_data = data.apply([](float x){ return std::log(x + 1e-8f); }); // epsilon for stability
    ValuePtr out = std::make_shared<Value>(out_data, std::vector<ValuePtr>{shared_from_this()}, "log");
    
    out->_backward = [this, out]() {
        // d(ln x)/dx = 1/x
        Matrix local_grad = data.apply([](float x){ return 1.0f / (x + 1e-8f); });
        this->grad.add(local_grad.hadamard(out->grad));
    };
    return out;
}

ValuePtr Value::neg() {
    Matrix out_data = data * -1.0f;
    ValuePtr out = std::make_shared<Value>(out_data, std::vector<ValuePtr>{shared_from_this()}, "neg");
    out->_backward = [this, out]() {
        Matrix neg = out->grad * -1.0f;
        this->grad.add(neg);
    };
    return out;
}

ValuePtr operator-(ValuePtr a, ValuePtr b) { return a->sub(b); }

// Division is just A * (B^-1)
ValuePtr operator/(ValuePtr a, ValuePtr b) { 
    return a->mul(b->pow(-1.0f)); 
}

ValuePtr operator+(ValuePtr a, ValuePtr b) { return a->add(b); }
ValuePtr operator*(ValuePtr a, ValuePtr b) { return a->matmul(b); }

ValuePtr Value::tanh() {
    // Forward: z = tanh(x)
    Matrix out_data = data.apply([](float x){ return std::tanh(x); });
    ValuePtr out = std::make_shared<Value>(out_data, std::vector<ValuePtr>{shared_from_this()}, "tanh");
    
    out->_backward = [this, out]() {
        // d(tanh)/dx = 1 - tanh^2
        // We can reuse 'out' since it holds tanh(x)
        Matrix local_grad = out->data.apply([](float t){ return 1.0f - (t * t); });
        this->grad.add(local_grad.hadamard(out->grad));
    };
    return out;
}

ValuePtr Value::sigmoid() {
    // Forward: z = 1 / (1 + e^-x)
    Matrix out_data = data.apply([](float x){ 
        return 1.0f / (1.0f + std::exp(-x)); 
    });
    ValuePtr out = std::make_shared<Value>(out_data, std::vector<ValuePtr>{shared_from_this()}, "sigmoid");
    
    out->_backward = [this, out]() {
        // d(sig)/dx = sig * (1 - sig)
        Matrix local_grad = out->data.apply([](float s){ return s * (1.0f - s); });
        this->grad.add(local_grad.hadamard(out->grad));
    };
    return out;
}

ValuePtr Value::softmax() {
    // 1. Forward Pass
    // We compute this row-wise (assuming x is 1xN or NxM)
    // For simplicity in this engine, we assume x is a SINGLE ROW vector (1xN).
    // (To support batches properly, we'd need a broadcasting engine, which is Phase V).
    
    // x_max for numerical stability (exp(x - max))
    float max_val = -1e9f;
    for(size_t i=0; i<data.cols; ++i) {
        if(data(0,i) > max_val) max_val = data(0,i);
    }
    
    Matrix out_data(data.rows, data.cols);
    float sum_exp = 0.0f;
    
    // Compute exp(x - max)
    for(size_t i=0; i<data.cols; ++i) {
        float e = std::exp(data(0,i) - max_val);
        out_data(0,i) = e;
        sum_exp += e;
    }
    
    // Normalize
    for(size_t i=0; i<data.cols; ++i) {
        out_data(0,i) /= sum_exp;
    }
    
    ValuePtr out = std::make_shared<Value>(out_data, std::vector<ValuePtr>{shared_from_this()}, "softmax");
    
    // 2. Backward Pass
    out->_backward = [this, out]() {
        // Gradient of Softmax:
        // dL/dx_i = S_i * (dL/dy_i - sum(S_k * dL/dy_k))
        
        // Compute dot product of (S . dL/dy)
        float s_dot_grad = 0.0f;
        for(size_t k=0; k<out->data.cols; ++k) {
            s_dot_grad += out->data(0,k) * out->grad(0,k);
        }
        
        for(size_t i=0; i<data.cols; ++i) {
            float s = out->data(0,i);
            float g = out->grad(0,i);
            
            // The formula simplifies nicely
            this->grad(0,i) += s * (g - s_dot_grad);
        }
    };
    
    return out;
}
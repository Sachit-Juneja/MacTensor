#include "../../include/dl/layers.h"
#include <cmath>
#include <random>

void Module::zero_grad() {
    for (auto p : parameters()) {
        for(size_t i=0; i<p->grad.rows; ++i)
            for(size_t j=0; j<p->grad.cols; ++j)
                p->grad(i,j) = 0.0f;
    }
}

Linear::Linear(int nin, int nout) {
    // Kaiming Init
    float scale = std::sqrt(2.0f / nin);
    Matrix w_data = Matrix::random(nin, nout);
    w_data.scale(scale);
    
    Matrix b_data(1, nout); // Bias 0

    W = Value::create(w_data);
    b = Value::create(b_data);
}

ValuePtr Linear::forward(ValuePtr x) {
    // y = xW + b
    return (x * W) + b;
}

std::vector<ValuePtr> Linear::parameters() {
    return {W, b};
}

ValuePtr ReLU::forward(ValuePtr x) {
    return x->relu();
}

std::vector<ValuePtr> ReLU::parameters() {
    return {};
}

Dropout::Dropout(float p) : p(p), training(true) {}

ValuePtr Dropout::forward(ValuePtr x) {
    if (!training) return x;
    
    float scale = 1.0f / (1.0f - p);
    
    Matrix mask_data(x->data.rows, x->data.cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution d(1.0f - p); 
    
    for(size_t i=0; i<mask_data.rows; ++i) {
        for(size_t j=0; j<mask_data.cols; ++j) {
            mask_data(i,j) = d(gen) ? scale : 0.0f;
        }
    }
    
    ValuePtr mask = Value::create(mask_data);
    return x->mul(mask);
}

std::vector<ValuePtr> Dropout::parameters() {
    return {}; // No parameters to train in Dropout
}
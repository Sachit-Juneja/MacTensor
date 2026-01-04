#include "../../include/dl/layers.h"
#include <cmath>

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
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
    // Kaiming Initialization (Scale by sqrt(2/nin))
    float scale = std::sqrt(2.0f / nin);
    Matrix w_data = Matrix::random(nin, nout);
    w_data.scale(scale);
    
    // Bias starts at 0
    Matrix b_data(1, nout); // 1xN row vector

    W = Value::create(w_data);
    b = Value::create(b_data);
}

ValuePtr Linear::forward(ValuePtr x) {
    // y = xW + b
    // Note: Our 'add' doesn't support broadcasting yet, so 'b' must match rows.
    // For now, let's assume batch_size=1 OR we manually broadcast in V2.
    // Hack: We rely on the fact that if x is 1xNin, xW is 1xNout, so + b (1xNout) works.
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
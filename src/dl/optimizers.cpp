#include "../../include/dl/optimizers.h"
#include <cmath>

Optimizer::Optimizer(std::vector<ValuePtr> params, float lr) : parameters(params), lr(lr) {}

void Optimizer::zero_grad() {
    for(auto p : parameters) {
        for(size_t i=0; i<p->grad.rows; ++i)
            for(size_t j=0; j<p->grad.cols; ++j)
                p->grad(i,j) = 0.0f;
    }
}

// --- SGD ---
SGD::SGD(std::vector<ValuePtr> params, float lr, float momentum) 
    : Optimizer(params, lr), momentum(momentum) {
    for(auto p : parameters) {
        velocities[p] = Matrix(p->data.rows, p->data.cols); // Init 0
    }
}

void SGD::step() {
    for(auto p : parameters) {
        // v = momentum * v - lr * grad
        Matrix& vel = velocities[p];
        
        // Scale momentum
        vel.scale(momentum); 
        
        // Subtract gradient term
        Matrix grad_term = p->grad * lr;
        vel.subtract(grad_term);
        
        // Update parameter: p += v
        p->data.add(vel);
    }
}

// --- Adam (Adaptive Moment Estimation) ---
Adam::Adam(std::vector<ValuePtr> params, float lr, float beta1, float beta2)
    : Optimizer(params, lr), beta1(beta1), beta2(beta2), epsilon(1e-8f), t(0) {
    
    for(auto p : parameters) {
        m[p] = Matrix(p->data.rows, p->data.cols); // Init 0
        v[p] = Matrix(p->data.rows, p->data.cols); // Init 0
    }
}

void Adam::step() {
    t++;
    
    for(auto p : parameters) {
        Matrix& m_t = m[p];
        Matrix& v_t = v[p];
        Matrix& g = p->grad;
        
        // 1. Update biased first moment estimate
        // m = beta1 * m + (1 - beta1) * g
        m_t.scale(beta1);
        Matrix g_scaled = g * (1.0f - beta1);
        m_t.add(g_scaled);
        
        // 2. Update biased second raw moment estimate
        // v = beta2 * v + (1 - beta2) * g^2
        v_t.scale(beta2);
        Matrix g_squared = g.hadamard(g);
        g_squared.scale(1.0f - beta2);
        v_t.add(g_squared);
        
        // 3. Compute bias-corrected moments
        // m_hat = m / (1 - beta1^t)
        float correction1 = 1.0f / (1.0f - std::pow(beta1, t));
        Matrix m_hat = m_t * correction1;
        
        // v_hat = v / (1 - beta2^t)
        float correction2 = 1.0f / (1.0f - std::pow(beta2, t));
        Matrix v_hat = v_t * correction2;
        
        // 4. Update parameters
        // p = p - lr * m_hat / (sqrt(v_hat) + eps)
        // We do the denominator element-wise
        Matrix update = m_hat.apply([this, &v_hat](float m_val){
            // This logic is tricky because we need corresponding v_hat val.
            // Since apply is element-wise but blind to index, we might need a loop here.
            return 0.0f; // placeholder
        });
        
        // Let's do a manual loop for the update step to be safe
        for(size_t i=0; i<p->data.rows; ++i) {
            for(size_t j=0; j<p->data.cols; ++j) {
                float mh = m_hat(i,j);
                float vh = v_hat(i,j);
                
                float delta = lr * mh / (std::sqrt(vh) + epsilon);
                p->data(i,j) -= delta;
            }
        }
    }
}
#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include "engine.h"
#include <vector>
#include <map>

class Optimizer {
public:
    std::vector<ValuePtr> parameters;
    float lr;

    Optimizer(std::vector<ValuePtr> params, float lr);
    virtual void step() = 0;
    void zero_grad();
};

class SGD : public Optimizer {
public:
    float momentum;
    std::map<ValuePtr, Matrix> velocities; // Store velocity for each param

    SGD(std::vector<ValuePtr> params, float lr = 0.01f, float momentum = 0.0f);
    void step() override;
};

class Adam : public Optimizer {
public:
    float beta1, beta2, epsilon;
    int t; // Timestep
    
    // First moment (m) and Second moment (v)
    std::map<ValuePtr, Matrix> m;
    std::map<ValuePtr, Matrix> v;

    Adam(std::vector<ValuePtr> params, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f);
    void step() override;
};

#endif
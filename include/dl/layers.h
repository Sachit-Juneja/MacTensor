#ifndef LAYERS_H
#define LAYERS_H

#include "engine.h"
#include <vector>

// Base class for all Neural Network modules
class Module {
public:
    virtual ~Module() = default;
    virtual ValuePtr forward(ValuePtr x) = 0;
    virtual std::vector<ValuePtr> parameters() = 0;
    void zero_grad(); // Reset gradients to 0
};

class Linear : public Module {
public:
    ValuePtr W;
    ValuePtr b;

    Linear(int nin, int nout);
    ValuePtr forward(ValuePtr x) override;
    std::vector<ValuePtr> parameters() override;
};

class ReLU : public Module {
public:
    ValuePtr forward(ValuePtr x) override;
    std::vector<ValuePtr> parameters() override;
};

// [NEW] Dropout Layer
// Randomly zeros out elements during training to prevent overfitting
class Dropout : public Module {
public:
    float p; // Probability of dropping a neuron
    bool training; // Only active during training
    
    Dropout(float p = 0.5f);
    ValuePtr forward(ValuePtr x) override;
    std::vector<ValuePtr> parameters() override;
};

// Wrappers for activations
class Tanh : public Module {
    ValuePtr forward(ValuePtr x) override { return x->tanh(); }
    std::vector<ValuePtr> parameters() override { return {}; }
};

class Sigmoid : public Module {
    ValuePtr forward(ValuePtr x) override { return x->sigmoid(); }
    std::vector<ValuePtr> parameters() override { return {}; }
};

#endif
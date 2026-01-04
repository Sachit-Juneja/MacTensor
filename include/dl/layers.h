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

#endif
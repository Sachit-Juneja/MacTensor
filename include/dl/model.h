#ifndef MODEL_H
#define MODEL_H

#include "layers.h"

class MLP : public Module {
public:
    std::vector<std::shared_ptr<Module>> layers;

    // e.g. sizes = {2, 16, 16, 1}
    MLP(int nin, std::vector<int> nouts);
    
    ValuePtr forward(ValuePtr x) override;
    std::vector<ValuePtr> parameters() override;
};

#endif
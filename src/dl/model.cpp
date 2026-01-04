#include "../../include/dl/model.h"

MLP::MLP(int nin, std::vector<int> nouts) {
    int sz_in = nin;
    for(size_t i=0; i<nouts.size(); ++i) {
        layers.push_back(std::make_shared<Linear>(sz_in, nouts[i]));
        sz_in = nouts[i];
        
        // Add ReLU between layers, but not after the last one
        if(i != nouts.size() - 1) {
            layers.push_back(std::make_shared<ReLU>());
        }
    }
}

ValuePtr MLP::forward(ValuePtr x) {
    for(auto layer : layers) {
        x = layer->forward(x);
    }
    return x;
}

std::vector<ValuePtr> MLP::parameters() {
    std::vector<ValuePtr> params;
    for(auto layer : layers) {
        auto p = layer->parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    return params;
}
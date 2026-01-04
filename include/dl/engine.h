#ifndef ENGINE_H
#define ENGINE_H

#include "../core/matrix.h"
#include <vector>
#include <functional>
#include <memory>
#include <unordered_set>

struct Value;
using ValuePtr = std::shared_ptr<Value>;

struct Value : public std::enable_shared_from_this<Value> {
    Matrix data;
    Matrix grad;
    
    // Graph connection: Who created me?
    std::vector<ValuePtr> children;
    
    // The "Chain Rule" function for this specific operation
    std::function<void()> _backward;
    
    std::string op; // Debug tag (e.g., "+", "matmul")

    // Constructor
    Value(Matrix data, std::vector<ValuePtr> children = {}, std::string op = "");

    // The Magic Button: Triggers backpropagation
    void backward();

    // Factory method for creating leaf nodes (inputs/weights)
    static ValuePtr create(Matrix d);

    // --- Operations ---
    ValuePtr add(ValuePtr other);
    ValuePtr matmul(ValuePtr other);
    ValuePtr relu();

    ValuePtr sub(ValuePtr other);
    ValuePtr mul(ValuePtr other); // Element-wise multiplication
    ValuePtr pow(float exponent);
    ValuePtr exp(); // e^x
    ValuePtr log(); // ln(x)
    ValuePtr neg(); // -x
};

// Operator Overloads for sugar (A + B)
ValuePtr operator+(ValuePtr a, ValuePtr b);
ValuePtr operator*(ValuePtr a, ValuePtr b); // Matrix Multiplication
ValuePtr operator-(ValuePtr a, ValuePtr b);
ValuePtr operator/(ValuePtr a, ValuePtr b); // a * b^-1

#endif
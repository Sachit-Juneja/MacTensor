#ifndef ENGINE_H
#define ENGINE_H

#include "../core/matrix.h"
#include <vector>
#include <functional>
#include <memory>
#include <unordered_set>

// Forward declaration
struct Value;

// We use shared_ptr because the graph structure is a messy web of ownership
// and we don't want to deal with double-free errors or memory leaks.
using ValuePtr = std::shared_ptr<Value>;

struct Value : public std::enable_shared_from_this<Value> {
    Matrix data;
    Matrix grad;
    
    // Who created me? (The Inputs)
    std::vector<ValuePtr> children;
    
    // How do I calculate gradients for my children?
    // This function is the "Chain Rule" stored as a closure
    std::function<void()> _backward;
    
    // Debugging label (optional)
    std::string op; 

    // Constructor: Wraps a matrix
    Value(Matrix data, std::vector<ValuePtr> children = {}, std::string op = "");

    // The Big Red Button: Triggers backpropagation
    void backward();

    // --- Operations (Factories) ---
    // These create NEW Value nodes and link them to the graph
    
    static ValuePtr create(Matrix d); // Leaf node creation

    ValuePtr add(ValuePtr other);
    ValuePtr matmul(ValuePtr other);
    ValuePtr relu();
    
    // Operator overloads for syntactic sugar (A + B)
    // defined outside usually, but we can do friends or helpers
};

// Operator Overloads for cleaner syntax
ValuePtr operator+(ValuePtr a, ValuePtr b);
ValuePtr operator*(ValuePtr a, ValuePtr b); // Matrix Mul (not elementwise for now)

#endif
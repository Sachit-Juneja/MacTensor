#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <Accelerate/Accelerate.h> // the secret sauce

class Matrix {
public:
    size_t rows;
    size_t cols;
    
    // storing flat data in a vector cause its easier to manage memory
    // standard vector is safer than raw pointers (no leaks pls)
    std::vector<float> data; 

    // constructors
    Matrix(size_t r, size_t c); 
    static Matrix random(size_t r, size_t c); // gaussian init
    static Matrix identity(size_t n);

    // accessors
    // IMPORTANT: using column-major order here!!
    // lapack throws a fit if i use row-major so we gotta adapt
    // index = col * rows + row
    float& operator()(size_t r, size_t c);
    const float& operator()(size_t r, size_t c) const;

    // math ops (wrappers around the cblas stuff)
    Matrix matmul(const Matrix& other) const; 
    
    // just prints the matrix so i can see if it worked
    void print() const;

    // vdsp vector operations
    // modifying in place cause copying memory is slow and expensive
    Matrix& add(const Matrix& other);      
    Matrix& subtract(const Matrix& other); 
    Matrix& scale(float scalar);           
    
    // operator overloads so i can write A + B instead of A.add(B)
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(float scalar) const;
};

#endif
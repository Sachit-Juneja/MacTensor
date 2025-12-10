#include "../include/core/matrix.h"
#include <iostream>

int main() {
    std::cout << "--- checking if mactensor works ---\n";

    // creating some random matrices to test
    // A is 2x3, B is 3x2. so result should be 2x2
    Matrix A = Matrix::random(2, 3);
    Matrix B = Matrix::random(3, 2);

    std::cout << "\nMatrix A:\n";
    A.print();
    
    std::cout << "\nMatrix B:\n";
    B.print();

    // trying out the optimized matmul
    // if this crashes, check cmake linkage
    Matrix C = A.matmul(B);

    std::cout << "\nResult C (A * B):\n";
    C.print();

    // simple identity matrix to start
    Matrix I = Matrix::identity(3);
    std::cout << "\nIdentity Matrix (I):\n";
    I.print();

    // random matrix for noise
    Matrix R = Matrix::random(3, 3);
    std::cout << "\nRandom Matrix (R):\n";
    R.print();

    // testing addition
    // effectively adding noise to identity
    Matrix Sum = R + I;
    std::cout << "\nSum (R + I):\n";
    Sum.print();

    // testing subtraction
    Matrix Diff = R - I;
    std::cout << "\nDifference (R - I):\n";
    Diff.print();

    // testing scalar multiplication
    // making the identity matrix big
    Matrix Scaled = I * 10.0f;
    std::cout << "\nScaled Identity (I * 10):\n";
    Scaled.print();

    return 0;
}
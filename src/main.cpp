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

    return 0;
}
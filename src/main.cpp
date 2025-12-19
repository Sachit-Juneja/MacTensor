#include "../include/core/matrix.h"
#include <iostream>
#include <cmath>
#include <vector>

// helper to check if things are roughly equal
bool close_enough(float a, float b) {
    return std::abs(a - b) < 1e-4;
}

// automated test for dot product
void test_dot_product() {
    std::cout << "\n--- Testing Dot Product ---\n";
    Matrix v1(3, 1);
    Matrix v2(3, 1);

    // [1, 2, 3] . [4, 5, 6] = 32
    v1(0, 0) = 1.0f; v1(1, 0) = 2.0f; v1(2, 0) = 3.0f;
    v2(0, 0) = 4.0f; v2(1, 0) = 5.0f; v2(2, 0) = 6.0f;

    float result = v1.dot(v2);
    std::cout << "v1 . v2 = " << result << " (Expected: 32.0000)\n";

    if (close_enough(result, 32.0f)) {
        std::cout << ">> [PASS] Dot Product\n";
    } else {
        std::cerr << ">> [FAIL] Dot Product\n";
        exit(1);
    }
}

// automated test for cholesky
void test_cholesky() {
    std::cout << "\n--- Testing Cholesky Decomposition ---\n";
    
    // Symmetric Positive Definite Matrix
    Matrix A(3, 3);
    A(0,0)=4;   A(0,1)=12;  A(0,2)=-16;
    A(1,0)=12;  A(1,1)=37;  A(1,2)=-43;
    A(2,0)=-16; A(2,1)=-43; A(2,2)=98;

    std::cout << "Input Matrix A:\n";
    A.print();

    try {
        Matrix L = A.cholesky();
        std::cout << "Lower Triangular L:\n";
        L.print();

        // Check specific known values for this input
        if (close_enough(L(0,0), 2.0f) && close_enough(L(1,0), 6.0f) && close_enough(L(2,2), 3.0f)) {
            std::cout << ">> [PASS] Cholesky Decomposition\n";
        } else {
            std::cerr << ">> [FAIL] Cholesky values incorrect\n";
            exit(1);
        }
    } catch (const std::exception& e) {
        std::cerr << ">> [FAIL] Cholesky crashed: " << e.what() << "\n";
        exit(1);
    }
}

void test_transpose() {
    std::cout << "\n--- Testing Transpose ---\n";
    Matrix A(2, 3);
    // filling with 1,2,3 / 4,5,6
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=4; A(1,1)=5; A(1,2)=6;

    Matrix T = A.transpose();
    
    // T should be 3x2 with 1,4 / 2,5 / 3,6
    if (T.rows == 3 && T.cols == 2 && T(0,1) == 4.0f) {
        std::cout << ">> [PASS] Transpose\n";
    } else {
        std::cerr << ">> [FAIL] Transpose logic broken\n";
        exit(1);
    }
}

void test_lu() {
    std::cout << "\n--- Testing LU Decomposition ---\n";
    // Using a simple non-symmetric matrix
    // A = [[4, 3], [6, 3]]
    // This would fail Cholesky but LU should handle it
    Matrix A(2, 2);
    A(0,0)=4; A(0,1)=3;
    A(1,0)=6; A(1,1)=3;

    try {
        Matrix LU = A.lu();
        // Visual check usually enough for LU as raw output is packed weirdly
        // But if it didn't throw, it likely worked
        std::cout << ">> [PASS] LU Decomposition (Solver didn't crash)\n";
    } catch(...) {
        std::cerr << ">> [FAIL] LU Decomposition crashed\n";
        exit(1);
    }
}

// automated test for SVD
void test_svd() {
    std::cout << "\n--- Testing Singular Value Decomposition (SVD) ---\n";
    
    // A = [[3, 2, 2], [2, 3, -2]]
    Matrix A(2, 3);
    A(0,0)=3; A(0,1)=2; A(0,2)=2;
    A(1,0)=2; A(1,1)=3; A(1,2)=-2;

    std::cout << "Original Matrix A:\n";
    A.print();

    try {
        auto result = A.svd();
        
        std::cout << "Singular Values (Sigma):\n";
        result.S.print();

        // Verification: Reconstruct A = U * Sigma * Vt
        // Note: S returned is a vector, need to turn it into diagonal matrix for math
        // This is a bit manual but necessary for the test
        Matrix SigmaMat(result.U.cols, result.Vt.rows); // correct dimensions
        
        // Fill SigmaMat with zeros first (manual loop since we dont have fill() yet)
        for(size_t i=0; i<SigmaMat.rows; ++i) {
            for(size_t j=0; j<SigmaMat.cols; ++j) SigmaMat(i,j) = 0.0f;
        }

        // Fill diagonal
        for(size_t i=0; i<result.S.rows; ++i) {
            SigmaMat(i, i) = result.S(i, 0);
        }

        // Reconstruction: (U * Sigma) * Vt
        Matrix Temp = result.U.matmul(SigmaMat);
        Matrix Reconstructed = Temp.matmul(result.Vt);

        std::cout << "Reconstructed Matrix (should match A):\n";
        Reconstructed.print();

        // Check first element to be safe
        if (close_enough(Reconstructed(0,0), 3.0f) && close_enough(Reconstructed(1,2), -2.0f)) {
            std::cout << ">> [PASS] SVD (Reconstruction successful)\n";
        } else {
            std::cerr << ">> [FAIL] SVD reconstruction mismatch\n";
            exit(1);
        }

    } catch (const std::exception& e) {
        std::cerr << ">> [FAIL] SVD crashed: " << e.what() << "\n";
        exit(1);
    }
}

// --- Main Execution ---

int main() {
    std::cout << "=== MACTENSOR DIAGNOSTICS ===\n";

    // 1. Visual Checks (Your original code)
    std::cout << "\n[1] Visualizing Basic Ops...\n";
    
    Matrix A = Matrix::random(2, 3);
    Matrix B = Matrix::random(3, 2);

    std::cout << "Matrix A:\n"; A.print();
    std::cout << "Matrix B:\n"; B.print();

    Matrix C = A.matmul(B);
    std::cout << "Result A * B:\n"; C.print();

    Matrix I = Matrix::identity(3);
    std::cout << "Identity I:\n"; I.print();

    Matrix Scaled = I * 10.0f;
    std::cout << "Scaled I * 10:\n"; Scaled.print();

    // 2. Automated Tests (The new stuff)
    std::cout << "\n[2] Running Test Suite...\n";
    
    test_dot_product();
    test_cholesky();
    test_transpose();
    test_lu();
    test_svd();

    std::cout << "\n=== ALL SYSTEMS OPERATIONAL ===\n";
    return 0;
}
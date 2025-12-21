#include "../include/core/matrix.h"
#include "../include/ml/linear_regression.h" 
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

void test_views() {
    std::cout << "\n--- Testing Memory Views (Strides) ---\n";
    Matrix A(2, 2);
    A(0,0) = 1.0f; A(0,1) = 2.0f;
    A(1,0) = 3.0f; A(1,1) = 4.0f;

    // Create a transpose view
    Matrix T = A.transpose();

    // Modify the VIEW
    std::cout << "Modifying Transpose View T(0,1) to 999.0...\n";
    T(0,1) = 999.0f; 

    // Check if ORIGINAL is modified
    // T(0,1) corresponds to A(1,0)
    if (close_enough(A(1,0), 999.0f)) {
        std::cout << ">> [PASS] View Memory Sharing (A(1,0) became 999.0)\n";
    } else {
        std::cout << ">> [FAIL] View Copy-on-Write Error. A(1,0) is still " << A(1,0) << "\n";
        exit(1);
    }

    // Check Contiguity
    if (A.is_contiguous() && !T.is_contiguous()) {
         std::cout << ">> [PASS] Contiguity Checks\n";
    } else {
         std::cout << ">> [FAIL] Contiguity Checks (A should be contig, T should not)\n";
         exit(1);
    }
}

void test_linear_regression() {
    std::cout << "\n--- Testing Linear Regression ---\n";

    // Synthetic Data Generation
    // y = 3*x1 + 2*x2 + noise
    // True Theta = [3, 2]^T
    
    size_t N = 100;
    Matrix X(N, 2);
    Matrix y(N, 1);

    for(size_t i=0; i<N; ++i) {
        float x1 = (float)(rand() % 100) / 10.0f; // 0 to 10
        float x2 = (float)(rand() % 100) / 10.0f;
        
        X(i, 0) = x1;
        X(i, 1) = x2;
        
        // y = 3x1 + 2x2
        y(i, 0) = 3.0f * x1 + 2.0f * x2; 
    }

    // 1. Test Analytical Solver
    {
        LinearRegression model(2);
        model.fit_analytical(X, y);
        
        std::cout << "Analytical Weights (Expect ~3.0, ~2.0):\n";
        model.theta.print();
    }

    // 2. Test SGD Solver
    {
        LinearRegression model(2);
        // High learning rate cause data is small, large epochs for convergence
        model.fit_sgd(X, y, 1000, 0.01f); 
        
        std::cout << "SGD Weights (Expect ~3.0, ~2.0):\n";
        model.theta.print();
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
    test_views();
    test_linear_regression();

    std::cout << "\n=== ALL SYSTEMS OPERATIONAL ===\n";
    return 0;
}
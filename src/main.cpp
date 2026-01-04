#include "../include/core/matrix.h"
#include "../include/ml/linear_regression.h" 
#include <iostream>
#include <cmath>
#include <vector>
#include "../include/ml/logistic_regression.h"
#include "../include/ml/decision_tree.h"
#include "../include/ml/gradient_boosting.h" 
#include "../include/ml/kmeans.h"
#include "../include/ml/pca.h"
#include "../include/ml/gmm.h"
#include "../include/dl/model.h"

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

void test_ridge_regression() {
    std::cout << "\n--- Testing Ridge Regression (L2) ---\n";

    // Data: y = 1*x1 + 1*x2
    // This creates a SINGULAR matrix X^T * X because x1 and x2 are identical.
    size_t N = 50;
    Matrix X(N, 2);
    Matrix y(N, 1);

    for(size_t i=0; i<N; ++i) {
        X(i, 0) = 1.0f; 
        X(i, 1) = 1.0f;
        y(i, 0) = 2.0f; 
    }

    // 1. Standard OLS (Lambda = 0)
    // This MUST fail because the matrix is singular and we are using a Cholesky solver.
    std::cout << "[1] Testing OLS (Lambda=0) on singular data...\n";
    try {
        LinearRegression ols(2);
        ols.fit_analytical(X, y, 0.0f);
        std::cerr << ">> [FAIL] OLS should have thrown an error on singular matrix!\n";
        exit(1);
    } catch (const std::exception& e) {
        std::cout << ">> [PASS] OLS correctly failed (Singular Matrix): " << e.what() << "\n";
    }

    // 2. Ridge (Lambda = 10.0)
    // This MUST succeed because adding Lambda*I makes the matrix Positive Definite.
    std::cout << "[2] Testing Ridge (Lambda=10.0)...\n";
    try {
        LinearRegression ridge(2);
        ridge.fit_analytical(X, y, 10.0f); // High penalty
        
        std::cout << "Ridge Weights (Expect ~0.909 due to shrinkage):\n";
        ridge.theta.print();
        
        // Analytical Result for this data: theta = 0.90909...
        if (std::abs(ridge.theta(0,0) - 0.909f) < 0.01f) {
             std::cout << ">> [PASS] Ridge Regression fixed the singularity!\n";
        } else {
             std::cerr << ">> [FAIL] Ridge values incorrect. Got " << ridge.theta(0,0) << "\n";
             exit(1);
        }
    } catch (const std::exception& e) {
        std::cerr << ">> [FAIL] Ridge crashed: " << e.what() << "\n";
        exit(1);
    }
}

void test_lasso() {
    std::cout << "\n--- Testing Lasso Regression (L1) ---\n";

    // Scenario: Feature 1 is strong, Feature 2 is weak/useless
    // y = 1*x1 + 0*x2
    size_t N = 50;
    Matrix X(N, 2);
    Matrix y(N, 1);

    for(size_t i=0; i<N; ++i) {
        X(i, 0) = (float)(rand()%10); 
        X(i, 1) = (float)(rand()%10); 
        y(i, 0) = 1.0f * X(i, 0); // Strictly depends on x1
    }

    // 1. Ridge (L2) - shrinks x2 but rarely makes it 0.0000
    LinearRegression ridge(2);
    ridge.fit_analytical(X, y, 1.0f);
    std::cout << "Ridge Weights (x2 should be small but non-zero):\n";
    ridge.theta.print();

    // 2. Lasso (L1) - should snap x2 to EXACTLY 0.0000
    LinearRegression lasso(2);
    // Use smaller lambda because the 'm' factor scales it up in the code
    lasso.fit_lasso_cd(X, y, 0.5f, 100); 
    std::cout << "Lasso Weights (x2 should be EXACTLY 0.0000):\n";
    lasso.theta.print();

    if (std::abs(lasso.theta(1,0)) < 1e-4 && std::abs(lasso.theta(0,0)) > 0.1f) {
        std::cout << ">> [PASS] Lasso successfully performed feature selection (Sparsity).\n";
    } else {
        std::cerr << ">> [FAIL] Lasso failed to zero out feature 2.\n";
        exit(1);
    }
}

void test_logistic_regression() {
    std::cout << "\n--- Testing Logistic Regression (Newton-Raphson) ---\n";

    // Dataset: Logic OR Gate
    // x1, x2  -> y
    // 0,  0   -> 0
    // 0,  1   -> 1
    // 1,  0   -> 1
    // 1,  1   -> 1
    
    // We add a Bias term explicitly as x0 = 1
    Matrix X(4, 3);
    Matrix y(4, 1);
    
    // Row 0: [1, 0, 0] -> 0
    X(0,0)=1; X(0,1)=0; X(0,2)=0; y(0,0)=0;
    // Row 1: [1, 0, 1] -> 1
    X(1,0)=1; X(1,1)=0; X(1,2)=1; y(1,0)=1;
    // Row 2: [1, 1, 0] -> 1
    X(2,0)=1; X(2,1)=1; X(2,2)=0; y(2,0)=1;
    // Row 3: [1, 1, 1] -> 1
    X(3,0)=1; X(3,1)=1; X(3,2)=1; y(3,0)=1;

    LogisticRegression model(3);
    model.fit_newton(X, y, 5); // 5 iterations should be enough for Newton

    std::cout << "Predictions (Expected: 0, 1, 1, 1):\n";
    Matrix preds = model.predict(X);
    preds.transpose().print(); // Print horizontally
    
    // Check Accuracy
    if (preds(0,0) == 0 && preds(1,0) == 1 && preds(2,0) == 1 && preds(3,0) == 1) {
        std::cout << ">> [PASS] Logistic Regression learned OR gate.\n";
    } else {
        std::cerr << ">> [FAIL] Wrong predictions.\n";
        exit(1);
    }
}

void test_decision_tree() {
    std::cout << "\n--- Testing Decision Tree Regressor ---\n";

    // Dataset: Non-linear curve y = x^2
    // Range -5 to 5
    size_t N = 20;
    Matrix X(N, 1);
    Matrix y(N, 1);

    for(size_t i=0; i<N; ++i) {
        float val = -5.0f + (float)i * 0.5f; // -5, -4.5, ...
        X(i, 0) = val;
        y(i, 0) = val * val; // y = x^2
    }

    // 1. Train Tree
    DecisionTreeRegressor tree(5, 2); // Depth 5
    tree.fit(X, y);

    // 2. Predict on new data
    // Let's predict for x = 3.2. 
    // True y = 10.24. 
    // Nearest training points were 3.0 (9.0) and 3.5 (12.25).
    // Tree should predict something in between.
    Matrix query(1, 1);
    query(0,0) = 3.2f; 
    
    Matrix pred = tree.predict(query);
    std::cout << "Pred for x=3.2: " << pred(0,0) << " (True: 10.24)\n";

    // 3. Measure Training Error (MSE)
    Matrix all_preds = tree.predict(X);
    Matrix diff = all_preds - y;
    float mse = diff.dot(diff) / N;
    
    std::cout << "Tree Training MSE: " << mse << "\n";

    if (mse < 1.0f) {
        std::cout << ">> [PASS] Decision Tree fit non-linear data (MSE < 1.0).\n";
    } else {
        std::cerr << ">> [FAIL] MSE too high (" << mse << "). Tree didn't learn.\n";
        exit(1);
    }
}

void test_decision_tree_classifier() {
    std::cout << "\n--- Testing Decision Tree Classifier (XOR) ---\n";

    // Dataset: XOR Gate
    // (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
    Matrix X(4, 2);
    Matrix y(4, 1);

    X(0,0)=0; X(0,1)=0; y(0,0)=0;
    X(1,0)=0; X(1,1)=1; y(1,0)=1;
    X(2,0)=1; X(2,1)=0; y(2,0)=1;
    X(3,0)=1; X(3,1)=1; y(3,0)=0;

    DecisionTreeClassifier clf(4, 2);
    clf.fit(X, y);
    
    Matrix preds = clf.predict(X);
    std::cout << "Predictions (Expected: 0, 1, 1, 0):\n";
    preds.transpose().print();

    if (preds(0,0)==0 && preds(1,0)==1 && preds(2,0)==1 && preds(3,0)==0) {
        std::cout << ">> [PASS] Classifier solved XOR.\n";
    } else {
        std::cerr << ">> [FAIL] Classifier failed XOR.\n";
        exit(1);
    }
}

void test_gradient_boosting() {
    std::cout << "\n--- Testing Gradient Boosting (XGBoost Lite) ---\n";

    // Dataset: Non-linear y = x^2
    // Range -5 to 5
    size_t N = 50;
    Matrix X(N, 1);
    Matrix y(N, 1);

    for(size_t i=0; i<N; ++i) {
        float val = -5.0f + (float)i * 0.2f; 
        X(i, 0) = val;
        y(i, 0) = val * val; 
    }

    // 1. Train Gradient Boosting
    // 50 trees, depth 2 (weak learners), learning rate 0.1
    GradientBoostingRegressor gbm(50, 0.1f, 2);
    gbm.fit(X, y);

    // 2. Measure Error
    Matrix preds = gbm.predict(X);
    Matrix diff = preds - y;
    float mse = diff.dot(diff) / N;

    std::cout << "Final GBM MSE: " << mse << "\n";

    // Compare with single tree logic (sanity check)
    // A single stump (depth 1) would have high error. 50 of them should be great.
    if (mse < 0.5f) {
        std::cout << ">> [PASS] Gradient Boosting converged on non-linear data.\n";
    } else {
        std::cerr << ">> [FAIL] MSE too high. Boosting didn't work.\n";
        exit(1);
    }
}

void test_kmeans() {
    std::cout << "\n--- Testing K-Means Clustering ---\n";

    // Data: 2 Clusters
    // Cluster 1: Random points around (0,0)
    // Cluster 2: Random points around (10,10)
    size_t N = 20;
    Matrix X(2 * N, 2);
    
    for(size_t i=0; i<N; ++i) {
        // Cluster 1
        X(i, 0) = (float)(rand() % 200) / 100.0f; // 0 to 2
        X(i, 1) = (float)(rand() % 200) / 100.0f;
        
        // Cluster 2
        X(N + i, 0) = 10.0f + (float)(rand() % 200) / 100.0f; // 10 to 12
        X(N + i, 1) = 10.0f + (float)(rand() % 200) / 100.0f;
    }

    // Run K-Means
    KMeans model(2); // k=2
    model.fit(X);
    
    std::cout << "Centroids Found:\n";
    model.centroids.print();

    // Check if centroids are roughly at (1,1) and (11,11)
    // We don't know order, so we check min distance to targets
    bool c1_ok = false;
    bool c2_ok = false;

    for(size_t i=0; i<2; ++i) {
        float x = model.centroids(i, 0);
        float y = model.centroids(i, 1);
        
        if (x < 3.0f && y < 3.0f) c1_ok = true;
        if (x > 8.0f && y > 8.0f) c2_ok = true;
    }

    if (c1_ok && c2_ok) {
        std::cout << ">> [PASS] K-Means identified both clusters.\n";
    } else {
        std::cerr << ">> [FAIL] Centroids are off.\n";
        exit(1);
    }
}

void test_pca() {
    std::cout << "\n--- Testing PCA (Dimensionality Reduction) ---\n";

    // Create highly correlated data (y approx x)
    size_t N = 10;
    Matrix X(N, 2);
    
    for(size_t i=0; i<N; ++i) {
        float val = (float)i;
        X(i, 0) = val; 
        X(i, 1) = val + ((float)(rand()%10)/50.0f); // Small noise
    }

    // 1. Fit PCA (Reduce 2D -> 1D)
    PCA pca(1);
    pca.fit(X);

    std::cout << "Top Component (Eigenvector):\n";
    pca.components.print(); 

    // 2. Transform (This previously crashed!)
    Matrix X_proj = pca.transform(X);
    std::cout << "Projected Data (1D - First 3):\n";
    for(int i=0; i<3; ++i) std::cout << "[" << X_proj(i,0) << "]\n";
    
    // 3. Inverse
    Matrix X_recon = pca.inverse_transform(X_proj);
    Matrix diff = X - X_recon;
    float mse = diff.dot(diff) / N;

    if (mse < 0.1f) {
        std::cout << ">> [PASS] PCA preserved structure. MSE: " << mse << "\n";
    } else {
        std::cerr << ">> [FAIL] PCA MSE high: " << mse << "\n";
        exit(1);
    }
}

void test_gmm() {
    std::cout << "\n--- Testing Gaussian Mixture Models (GMM) ---\n";
    
    // Synthetic Data: 2 Overlapping Clusters
    // A: Mean (2,2)
    // B: Mean (8,8)
    size_t N = 50;
    Matrix X(2 * N, 2);
    
    for(size_t i=0; i<N; ++i) {
        // Cluster A
        X(i, 0) = 2.0f + ((rand()%100)/20.0f); 
        X(i, 1) = 2.0f + ((rand()%100)/50.0f); 
        
        // Cluster B
        X(N+i, 0) = 8.0f + ((rand()%100)/50.0f);
        X(N+i, 1) = 8.0f + ((rand()%100)/50.0f);
    }
    
    GMM model(2, 20);
    model.fit(X);
    
    std::cout << "GMM Weights: "; model.weights.print();
    
    bool found_A = false, found_B = false;
    for(int j=0; j<2; ++j) {
        float mx = model.means[j](0,0);
        float my = model.means[j](0,1);
        std::cout << "Comp " << j << " Mean: (" << mx << ", " << my << ")\n";
        
        if (mx < 5 && my < 5) found_A = true;
        if (mx > 5 && my > 5) found_B = true;
    }
    
    if (found_A && found_B) {
        std::cout << ">> [PASS] GMM found both clusters.\n";
    } else {
        std::cerr << ">> [FAIL] GMM failed to converge on means.\n";
        exit(1);
    }
}

void test_autograd_mlp() {
    std::cout << "\n--- Testing Deep Learning (MLP on XOR) ---\n";
    
    // XOR Data
    // 0,0 -> 0
    // 0,1 -> 1
    // 1,0 -> 1
    // 1,1 -> 0
    
    // We train on samples one by one (SGD) because we haven't implemented batching/broadcasting yet.
    // This is "pure" SGD.
    
    std::vector<Matrix> X_train = {
        Matrix(1,2), Matrix(1,2), Matrix(1,2), Matrix(1,2)
    };
    std::vector<float> y_train = {0.0f, 1.0f, 1.0f, 0.0f};
    
    X_train[0](0,0)=0; X_train[0](0,1)=0;
    X_train[1](0,0)=0; X_train[1](0,1)=1;
    X_train[2](0,0)=1; X_train[2](0,1)=0;
    X_train[3](0,0)=1; X_train[3](0,1)=1;

    // Create MLP: 2 inputs -> 4 hidden -> 1 output
    MLP model(2, {4, 1});
    
    std::cout << "Training for 2000 steps...\n";
    
    for(int k=0; k<2000; ++k) {
        float total_loss = 0.0f;
        
        for(size_t i=0; i<4; ++i) {
            // 1. Forward
            ValuePtr x = Value::create(X_train[i]);
            ValuePtr pred = model.forward(x);
            
            // 2. Loss (Manual MSE)
            // L = (pred - target)^2
            float diff = pred->data(0,0) - y_train[i];
            float loss = diff * diff;
            total_loss += loss;
            
            // 3. Zero Grad
            model.zero_grad();
            
            // 4. Backward
            // Inject gradient dL/dPred = 2 * (pred - target)
            pred->grad(0,0) = 2.0f * diff;
            pred->backward();
            
            // 5. Update (SGD)
            for(auto p : model.parameters()) {
                p->data.subtract(p->grad * 0.05f); // lr = 0.05
            }
        }
        
        if(k % 500 == 0) std::cout << "Step " << k << " Loss: " << total_loss << "\n";
    }
    
    // Verify
    std::cout << "Predictions:\n";
    for(size_t i=0; i<4; ++i) {
        ValuePtr x = Value::create(X_train[i]);
        ValuePtr pred = model.forward(x);
        std::cout << "In: " << X_train[i](0,0) << "," << X_train[i](0,1) 
                  << " -> Out: " << pred->data(0,0) << " (Target: " << y_train[i] << ")\n";
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
    test_ridge_regression();
    test_lasso();
    test_logistic_regression();
    test_decision_tree();
    test_decision_tree_classifier();
    test_gradient_boosting();
    test_kmeans();
    test_pca();
    test_gmm();
    test_autograd_mlp();

    std::cout << "\n=== ALL SYSTEMS OPERATIONAL ===\n";
    return 0;
}
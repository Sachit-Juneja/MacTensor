# MacTensor: A High-Performance Computational Framework for Machine Learning

**MacTensor** is a monolithic library written in C++17 that implements the complete machine learning stack from first principles.

The primary objective of this project is to bridge the gap between abstract mathematical theory (Applied Mathematics, Optimization) and low-level systems engineering. Unlike high-level wrappers, MacTensor rebuilds the fundamental infrastructure of modern AI—from memory management and linear algebra kernels to automatic differentiation and convex optimization solvers.

This library is explicitly engineered for **macOS Apple Silicon (M1/M2/M3)** architectures, leveraging the Apple Accelerate Framework to utilize the AMX co-processor for high-throughput matrix operations.

---

## 1. System Architecture

The library follows a strict three-layer hierarchical architecture. Each layer abstracts the complexity of the layer below it.

[Image of layered software architecture diagram showing Hardware, Core Matrix, Classical ML, and Deep Learning layers]

1.  **Layer I (Core):** The Linear Algebra foundation. Wraps Apple Accelerate (BLAS/LAPACK) and handles memory management.
2.  **Layer II (ML):** The Statistical Optimization Engine. Implements classical algorithms (Regression, Clustering) using closed-form and iterative solvers.
3.  **Layer III (DL):** The Deep Learning Engine. Implements a reverse-mode Autograd system on a dynamic computational graph.

---

## 2. Implementation Roadmap & Checklist

### Phase I: Mathematical Foundation (Linear Algebra Core)
*The numerical engine that powers the entire library.*

- [x] **Matrix Class Architecture**
    - [x] Implement `Matrix` class with `std::vector` storage.
    - [x] Implement **Column-Major** storage logic (required for LAPACK compatibility).
    - [x] Implement Strides and Views for non-contiguous memory access.
- [x] **Apple Accelerate Integration (BLAS)**
    - [x] Link `Accelerate.framework` via CMake.
    - [x] Implement `matmul` using `cblas_sgemm` (Level 3 BLAS).
    - [x] Implement vector operations using `vDSP` (Addition, Scaling, Dot Product).
- [x] **Linear Solvers (LAPACK)**
    - [x] Implement Cholesky Decomposition (`spotrf`) for symmetric positive-definite matrices.
    - [x] Implement LU Decomposition (`sgetrf`) for general systems.
    - [x] Implement Singular Value Decomposition (`sgesvd`) for dimensionality reduction.

### Phase II: Classical Supervised Learning
*Optimization problems mapping Input X to Output Y.*

- [x] **Linear Regression**
    - [x] Analytical Solver: Normal Equation $\theta = (X^T X)^{-1} X^T y$ using Cholesky.
    - [x] Iterative Solver: Stochastic Gradient Descent (SGD) implementation.
- [x] **Regularization**
    - [x] Ridge Regression (L2 Penalty): Modification of the Normal Equation.
    - [x] Lasso Regression (L1 Penalty): Coordinate Descent implementation.
- [x] **Logistic Regression**
    - [x] Sigmoid Activation Function $\sigma(z) = 1 / (1 + e^{-z})$.
    - [x] Cross-Entropy Loss (Log-Loss) calculation.
    - [x] Newton-Raphson optimization method (utilizing the Hessian).
- [ ] **Tree-Based Models**
    - [ ] Decision Tree Regressor (Variance Reduction split).
    - [ ] Decision Tree Classifier (Gini Impurity / Entropy split).
    - [ ] **XGBoost Implementation:** Gradient Boosting on Decision Trees (Additive Training).

### Phase III: Unsupervised Learning
*Pattern discovery and dimensionality reduction.*

- [ ] **Clustering**
    - [ ] K-Means: Lloyd's Algorithm implementation.
    - [ ] K-Means++: Probabilistic seeding for faster convergence.
- [ ] **Dimensionality Reduction**
    - [ ] Principal Component Analysis (PCA) via Covariance Matrix decomposition.
    - [ ] PCA via SVD (for numerical stability).
- [ ] **Gaussian Mixture Models**
    - [ ] Expectation-Maximization (EM) algorithm implementation.

### Phase IV: Deep Learning (Autograd & Neural Networks)
*Differentiable programming on a dynamic graph.*

- [ ] **Automatic Differentiation Engine**
    - [ ] `Value` node wrapper (stores data, gradient, and backward function).
    - [ ] Topological Sort algorithm for Graph Traversal.
    - [ ] **Backward Ops:**
        - [ ] Addition / Subtraction / Multiplication.
        - [ ] Matrix Multiplication (Transpose logic).
        - [ ] Power / Exponentiation / Logarithm.
- [ ] **Neural Network Modules**
    - [ ] `Module` base class with parameter registration.
    - [ ] `Linear` (Dense) Layer implementation.
    - [ ] `Dropout` Layer implementation.
- [ ] **Activation Functions**
    - [ ] ReLU (Rectified Linear Unit).
    - [ ] Softmax (with LogSumExp stability trick).
    - [ ] Tanh / Sigmoid.
- [ ] **Optimizers**
    - [ ] SGD with Momentum.
    - [ ] Adam (Adaptive Moment Estimation).

---

## 3. Mathematical Theory

### Automatic Differentiation (Reverse Mode)
MacTensor utilizes reverse-mode AD. Given a scalar loss $L$ and an intermediate variable $z = f(x, y)$, gradients are computed recursively:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial x}$$

### Convex Optimization
For linear models, we solve the optimization problem:

$$\min_{\theta} \frac{1}{2m} || X\theta - y ||^2_2 + \lambda || \theta ||_p$$

This is solved efficiently using LAPACK's matrix factorization routines rather than purely iterative methods, demonstrating the advantage of a C++ backend.

---

## 4. Build Instructions

### Prerequisites
* **OS:** macOS (Apple Silicon recommended).
* **Compiler:** Clang (C++17 standard).
* **Build System:** CMake 3.10+.

### Compilation
```bash
# 1. Clone
git clone https://github.com/Sachit-Juneja/MacTensor.git
cd MacTensor

# 2. Build
mkdir build && cd build
cmake ..
make -j4
```
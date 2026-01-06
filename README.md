# MacTensor: A High-Performance Computational Framework for Machine Learning

**MacTensor** is a monolithic library written in C++17 that implements the complete machine learning stack from first principles.

The primary objective of this project is to bridge the gap between abstract mathematical theory (Applied Mathematics, Optimization) and low-level systems engineering. Unlike high-level wrappers, MacTensor rebuilds the fundamental infrastructure of modern AI—from memory management and linear algebra kernels to automatic differentiation and convex optimization solvers.

This library is explicitly engineered for **macOS Apple Silicon (M1/M2/M3)** architectures, leveraging the Apple Accelerate Framework to utilize the AMX co-processor for high-throughput matrix operations.

---

## 1. System Architecture

The library follows a strict four-layer hierarchical architecture. Each layer abstracts the complexity of the layer below it.

1.  **Layer I (Core):** The Linear Algebra foundation. Wraps Apple Accelerate (BLAS/LAPACK) and handles memory management via smart pointers and strided views.
2.  **Layer II (ML):** The Statistical Optimization Engine. Implements classical algorithms (Regression, Clustering, Trees) using closed-form and iterative solvers.
3.  **Layer III (Unsupervised):** Pattern discovery engine implementing clustering (K-Means), density estimation (GMM), and dimensionality reduction (PCA).
4.  **Layer IV (DL):** The Deep Learning Engine. Implements a reverse-mode Autograd system on a dynamic computational graph with standard layers and optimizers.

---

## 2. Implementation Status & Roadmap

### ✅ Phase I: Mathematical Foundation (Linear Algebra Core)
*The numerical engine that powers the entire library.*
- [x] **Matrix Class Architecture:** `std::shared_ptr` storage, Column-Major logic, Strides/Views.
- [x] **Apple Accelerate Integration:** `cblas_sgemm` (MatMul), `vDSP` (Vector Ops).
- [x] **Linear Solvers (LAPACK):** Cholesky (`spotrf`), LU (`sgetrf`), SVD (`sgesvd`), Inverse/Determinant.

### ✅ Phase II: Classical Supervised Learning
*Optimization problems mapping Input X to Output Y.*
- [x] **Linear Regression:** Analytical (Normal Equation) & Iterative (SGD).
- [x] **Regularization:** Ridge (L2) & Lasso (Coordinate Descent).
- [x] **Logistic Regression:** Newton-Raphson Solver.
- [x] **Tree-Based Models:** Decision Tree Regressor/Classifier, Gradient Boosting (XGBoost logic).

### ✅ Phase III: Unsupervised Learning
*Pattern discovery and dimensionality reduction.*
- [x] **Clustering:** K-Means (Lloyd's Algorithm) with K-Means++ Initialization.
- [x] **Dimensionality Reduction:** PCA via Singular Value Decomposition (SVD).
- [x] **Density Estimation:** Gaussian Mixture Models (GMM) via Expectation-Maximization.

### ✅ Phase IV: Deep Learning (Autograd & Neural Networks)
*Differentiable programming on a dynamic graph.*
- [x] **Autograd Engine:** Reverse-mode automatic differentiation (Topological Sort).
- [x] **Math Operations:** `add`, `sub`, `mul`, `matmul`, `pow`, `exp`, `log`, `tanh`, `sigmoid`, `softmax`.
- [x] **Layers:** `Linear` (Dense), `ReLU`, `Dropout`, `Tanh`, `Sigmoid`.
- [x] **Optimizers:** `SGD` (with Momentum), `Adam` (Adaptive Moment Estimation).
- [x] **Loss Functions:** MSE (Manual), Cross-Entropy (via Softmax-Logits).

### 🚧 Phase V: Future Work (Production & Scale)
*Features required to scale from educational prototype to production library.*
- [ ] **Mini-Batching & Broadcasting:** Support for `(Batch, N)` operations in the Matrix class to maximize throughput.
- [ ] **Convolutional Layers:** Implement `Conv2d` using `im2col` + GEMM for Computer Vision.
- [ ] **Model Serialization:** Save/Load model weights to binary files.
- [ ] **GPU Support:** Port the BLAS backend to Apple Metal Performance Shaders (MPS) for massive matrix operations.

---

## 3. When to use MacTensor vs. PyTorch

MacTensor is not designed to replace PyTorch for training LLMs or massive ResNets. It fills a specific niche for embedded, low-dependency, and educational use cases on macOS.

| Feature | **MacTensor (C++)** | **PyTorch (Python/LibTorch)** |
| :--- | :--- | :--- |
| **Use Case** | Native macOS apps, Audio Plugins, Embedded Logic, Learning Internals. | Research, Large Scale Training, Server-side Inference. |
| **Overhead** | **Minimal.** Zero Python interpreter overhead. Tiny binary size (<1MB). Instant startup. | **Heavy.** Requires Python runtime or large LibTorch dylibs (>1GB). Slow cold start. |
| **Dependencies** | **Zero.** Depends only on system `Accelerate.framework`. | **Many.** Requires Python, NumPy, etc. |
| **Performance** | **High (CPU/AMX).** Optimized for Apple Silicon CPU. Great for small-to-medium models. | **Extreme (GPU).** Uses Metal/CUDA. Faster for massive matrices (>2048x2048). |

**Choose MacTensor if:**
* You are building a standalone macOS desktop application (e.g., C++ JUCE plugin) and cannot bundle a 1GB Python environment.
* You need instant startup times and low memory footprint.
* You want to understand *exactly* how `backward()` works without digging through millions of lines of code.

**Choose PyTorch if:**
* You are training large-scale models (Transformers, Diffusion models).
* You need CUDA/Nvidia support.
* You rely on the vast ecosystem of pre-trained models (HuggingFace).

---

## 4. Mathematical Theory

### Automatic Differentiation (Reverse Mode)
MacTensor utilizes a dynamic computational graph. Given a scalar loss $L$ and an intermediate variable $z = f(x, y)$, gradients are computed recursively via the Chain Rule:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial x}$$

### Convex Optimization
For linear models, we solve the optimization problem using LAPACK's matrix factorization routines rather than purely iterative methods, demonstrating the advantage of a C++ backend:

$$\min_{\theta} \frac{1}{2m} || X\theta - y ||^2_2 + \lambda || \theta ||_p$$

---

## 5. Build Instructions

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

# 3. Run Diagnostics
./MacTensor
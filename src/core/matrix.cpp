#include "../../include/core/matrix.h"
#include <random>
#include <iomanip>

// init with zeros
Matrix::Matrix(size_t r, size_t c) : rows(r), cols(c) {
    // just resizing vector, fills with 0 by default
    data.resize(r * c, 0.0f);
}

// accessing elements
// this was annoying to figure out. standard c++ is row-major
// but accelerate wants col-major. 
// so M(row, col) is actually data[col * rows + row]
float& Matrix::operator()(size_t r, size_t c) {
    return data[c * rows + r];
}

const float& Matrix::operator()(size_t r, size_t c) const {
    return data[c * rows + r];
}

// random gaussian noise (mean=0, var=1)
// good for initializing weights later
Matrix Matrix::random(size_t r, size_t c) {
    Matrix m(r, c);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(0.0f, 1.0f);

    for(auto& val : m.data) {
        val = d(gen);
    }
    return m;
}

// the big one. matrix multiplication using apple's hardware acceleration
Matrix Matrix::matmul(const Matrix& B) const {
    // sanity check dimensions
    if (cols != B.rows) {
        throw std::invalid_argument("dimensions dont match bro");
    }

    Matrix C(rows, B.cols);

    // calling cblas_sgemm (single precision general matrix multiply)
    // this function signature is massive, had to read docs carefully
    cblas_sgemm(
        CblasColMajor,      // tell it we are using col-major storage
        CblasNoTrans,       // dont transpose A
        CblasNoTrans,       // dont transpose B
        (int)rows,          // M
        (int)B.cols,        // N
        (int)cols,          // K
        1.0f,               // alpha (scaling factor)
        data.data(),        // pointer to A
        (int)rows,          // lda (leading dimension of A)
        B.data.data(),      // pointer to B
        (int)B.rows,        // ldb
        0.0f,               // beta (scaling for C, 0 means overwrite)
        C.data.data(),      // pointer to C
        (int)rows           // ldc
    );

    return C;
}

// helper to visualize what's happening
void Matrix::print() const {
    std::cout << "Matrix(" << rows << "x" << cols << "):\n";
    for(size_t i = 0; i < rows; ++i) {
        std::cout << "[ ";
        for(size_t j = 0; j < cols; ++j) {
            // printing nicely with 4 decimal places
            std::cout << std::fixed << std::setprecision(4) << (*this)(i, j) << " ";
        }
        std::cout << "]\n";
    }
}

// identity matrix implementation
// just ones on the diagonal, zeros everywhere else
Matrix Matrix::identity(size_t n) {
    Matrix m(n, n);
    for(size_t i = 0; i < n; ++i) {
        m(i, i) = 1.0f;
    }
    return m;
}

// element wise addition using apple vdsp
// basically utilizing the simd registers to go fast
Matrix& Matrix::add(const Matrix& other) {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("cant add matrices with different shapes");
    }

    // vdsp_vadd adds B + A into C
    // confusing argument order but whatever
    vDSP_vadd(
        data.data(), 1,         // this is A
        other.data.data(), 1,   // this is B
        data.data(), 1,         // result goes back into A
        data.size()             
    );
    
    return *this;
}

// element wise subtraction
Matrix& Matrix::subtract(const Matrix& other) {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("shapes dont match for subtraction");
    }

    // formula is C = A - B
    // pass other as B, this as A
    vDSP_vsub(
        other.data.data(), 1,   
        data.data(), 1,         
        data.data(), 1,         
        data.size()
    );

    return *this;
}

// scalar multiplication
// nice for learning rates later
Matrix& Matrix::scale(float scalar) {
    vDSP_vsmul(
        data.data(), 1,
        &scalar,
        data.data(), 1,
        data.size()
    );

    return *this;
}

// operator overloads
// these create copies cause sometimes we want A + B to be a new matrix
Matrix Matrix::operator+(const Matrix& other) const {
    Matrix result = *this; 
    result.add(other);     
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    Matrix result = *this;
    result.subtract(other);
    return result;
}

Matrix Matrix::operator*(float scalar) const {
    Matrix result = *this;
    result.scale(scalar);
    return result;
}

// dot product
// basically measures how much two vectors point in the same direction
float Matrix::dot(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("shapes must match for dot product");
    }

    float result = 0.0f;
    
    // vdsp_dotpr goes brrr
    // stride is 1 cause our data is contiguous
    vDSP_dotpr(
        data.data(), 1, 
        other.data.data(), 1, 
        &result, 
        data.size()
    );

    return result;
}

// cholesky decomposition (LL^T)
// using lapack spotrf. this modifies the matrix in place usually
// but we return a new L lower triangular matrix
Matrix Matrix::cholesky() const {
    if (rows != cols) {
        throw std::invalid_argument("cholesky requires square matrix");
    }

    // copy current matrix cause lapack destroys the input
    Matrix L = *this;

    int n = (int)rows;
    int lda = n;
    int info = 0;
    
    // calling the lapack routine directly
    // "L" means fill the lower triangle. 
    // careful: lapack expects mutable pointers
    spotrf_("L", &n, L.data.data(), &lda, &info);

    if (info != 0) {
        throw std::runtime_error("cholesky failed. matrix might not be positive definite :(");
    }

    // spotrf leaves garbage in the upper triangle, so we gotta zero it out
    // strictly speaking we should only return the lower triangle
    for(size_t i = 0; i < rows; ++i) {
        for(size_t j = i + 1; j < cols; ++j) {
            L(i, j) = 0.0f;
        }
    }

    return L;
}

// simple transpose. standard O(n^2) but needed everywhere
Matrix Matrix::transpose() const {
    Matrix T(cols, rows); // dimensions flipped
    for(size_t i = 0; i < rows; ++i) {
        for(size_t j = 0; j < cols; ++j) {
            T(j, i) = (*this)(i, j);
        }
    }
    return T;
}

// LU decomposition using lapack sgetrf
// this is the bread and butter solver for general systems
Matrix Matrix::lu() const {
    if (rows != cols) {
        throw std::invalid_argument("lu decomp requires square matrix");
    }

    Matrix Result = *this; // copy data
    
    int n = (int)rows;
    int lda = n;
    int info = 0;
    
    // pivot indices (records row swaps)
    // lapack needs this to solve the system later
    std::vector<int> ipiv(n);

    // calculates PLU factorization
    // result is stored in-place:
    // L is below diagonal (unit diagonal implied)
    // U is above diagonal
    sgetrf_(&n, &n, Result.data.data(), &lda, ipiv.data(), &info);

    if (info != 0) {
        throw std::runtime_error("lu decomposition failed. matrix is singular (uninvertible)");
    }

    return Result;
}

// singular value decomposition
// the heavy lifter for dimensionality reduction
Matrix::SVDResult Matrix::svd() const {
    Matrix U(rows, rows);
    Matrix Vt(cols, cols);
    Matrix S(std::min(rows, cols), 1); // singular values are a vector really

    // lapack workspace query
    // sgesvd requires a work array, we ask it how much memory it needs first
    int m = (int)rows;
    int n = (int)cols;
    int lda = m;
    int ldu = m;
    int ldvt = n;
    int info = 0;
    float wkopt;
    int lwork = -1; // query mode

    // first call to get optimal workspace size
    sgesvd_("A", "A", &m, &n, nullptr, &lda, nullptr, nullptr, &ldu, nullptr, &ldvt, &wkopt, &lwork, &info);
    
    // allocate the workspace
    lwork = (int)wkopt;
    std::vector<float> work(lwork);

    // actual calculation
    // "A" means return all columns of U and VT
    // passing data copies because sgesvd destroys the input matrix
    std::vector<float> a_copy = data;
    
    sgesvd_(
        "A", "A", 
        &m, &n, 
        a_copy.data(), &lda, 
        S.data.data(), 
        U.data.data(), &ldu, 
        Vt.data.data(), &ldvt, 
        work.data(), &lwork, 
        &info
    );

    if (info > 0) {
        throw std::runtime_error("svd failed to converge");
    }

    return {U, S, Vt};
}
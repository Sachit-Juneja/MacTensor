#include "../../include/core/matrix.h"
#include <random>
#include <iomanip>

// init with zeros
Matrix::Matrix(size_t r, size_t c) 
    : rows(r), cols(c), offset(0), stride_rows(1), stride_cols(r) {
    // just resizing vector, fills with 0 by default
    // [modified]: allocating shared storage
    data = std::make_shared<std::vector<float>>(r * c, 0.0f);
}

// [new]: view constructor
Matrix::Matrix(size_t r, size_t c, std::shared_ptr<std::vector<float>> ptr, size_t off, size_t str_r, size_t str_c)
    : rows(r), cols(c), data(ptr), offset(off), stride_rows(str_r), stride_cols(str_c) {}

// [new]: helpers
bool Matrix::is_contiguous() const {
    return stride_rows == 1 && stride_cols == rows;
}

float* Matrix::raw_data() const {
    return data->data() + offset;
}

// accessing elements
// this was annoying to figure out. standard c++ is row-major
// but accelerate wants col-major. 
// so M(row, col) is actually data[col * rows + row]
// [modified]: updated for general stride formula
float& Matrix::operator()(size_t r, size_t c) {
    return (*data)[offset + c * stride_cols + r * stride_rows];
}

const float& Matrix::operator()(size_t r, size_t c) const {
    return (*data)[offset + c * stride_cols + r * stride_rows];
}

// random gaussian noise (mean=0, var=1)
// good for initializing weights later
Matrix Matrix::random(size_t r, size_t c) {
    Matrix m(r, c);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(0.0f, 1.0f);

    // [modified]: iterate directly over storage for speed
    for(auto& val : *m.data) {
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
    CBLAS_TRANSPOSE TransA = CblasNoTrans;
    CBLAS_TRANSPOSE TransB = CblasNoTrans;
    
    int lda = (int)stride_cols;
    int ldb = (int)B.stride_cols;

    // if stride_rows > 1, it's effectively transposed (row major)
    // so we cheat and tell BLAS to transpose it
    if (stride_rows > 1 && stride_cols == 1) {
        TransA = CblasTrans;
        lda = (int)stride_rows;
    }
    
    if (B.stride_rows > 1 && B.stride_cols == 1) {
        TransB = CblasTrans;
        ldb = (int)B.stride_rows;
    }

    // [FIX]: BLAS throws a tantrum if leading dimension is too small.
    // Even if it's a 10x1 vector with stride 1, BLAS demands lda >= 10.
    // So if we are "NoTrans", ensure stride satisfies the dimension requirement.
    if (TransA == CblasNoTrans && lda < rows) lda = (int)rows;
    if (TransB == CblasNoTrans && ldb < B.rows) ldb = (int)B.rows;

    cblas_sgemm(
        CblasColMajor,      // tell it we are using col-major storage
        TransA,             // transpose A if needed
        TransB,             // transpose B if needed
        (int)rows,          // M
        (int)B.cols,        // N
        (int)cols,          // K
        1.0f,               // alpha (scaling factor)
        raw_data(),         // pointer to A
        lda,                // lda (leading dimension of A)
        B.raw_data(),       // pointer to B
        ldb,                // ldb
        0.0f,               // beta (scaling for C, 0 means overwrite)
        C.raw_data(),       // pointer to C
        (int)rows           // ldc
    );

    return C;
}

// helper to visualize what's happening
void Matrix::print() const {
    std::cout << "Matrix(" << rows << "x" << cols << ")";
    if (!is_contiguous()) std::cout << " [View]";
    std::cout << ":\n";
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
    
    // [modified]: check if we can use fast vdsp (must be contiguous)
    if (is_contiguous() && other.is_contiguous()) {
        vDSP_vadd(
            raw_data(), 1,         // this is A
            other.raw_data(), 1,   // this is B
            raw_data(), 1,         // result goes back into A
            rows * cols             
        );
    } else {
        // slow path for complex views
        for(size_t i=0; i<rows; ++i)
            for(size_t j=0; j<cols; ++j)
                (*this)(i,j) += other(i,j);
    }
    
    return *this;
}

// element wise subtraction
Matrix& Matrix::subtract(const Matrix& other) {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("shapes dont match for subtraction");
    }

    // formula is C = A - B
    // pass other as B, this as A
    if (is_contiguous() && other.is_contiguous()) {
        vDSP_vsub(
            other.raw_data(), 1,   
            raw_data(), 1,         
            raw_data(), 1,         
            rows * cols
        );
    } else {
        // slow path
        for(size_t i=0; i<rows; ++i)
            for(size_t j=0; j<cols; ++j)
                (*this)(i,j) -= other(i,j);
    }

    return *this;
}

// scalar multiplication
// nice for learning rates later
Matrix& Matrix::scale(float scalar) {
    if (is_contiguous()) {
        vDSP_vsmul(
            raw_data(), 1,
            &scalar,
            raw_data(), 1,
            rows * cols
        );
    } else {
        for(size_t i=0; i<rows; ++i)
            for(size_t j=0; j<cols; ++j)
                (*this)(i,j) *= scalar;
    }

    return *this;
}

// operator overloads
// these create copies cause sometimes we want A + B to be a new matrix
Matrix Matrix::operator+(const Matrix& other) const {
    Matrix result = *this; // [note]: if this is a view, result is a view too
    
    // if we are a view, we need a deep copy before modifying to avoid side effects
    // but for now let's assume result follows copy-on-write or just standard copy
    // actually, to be safe with shared_ptr, we should clone if we want a new matrix
    if (!is_contiguous()) {
        // force a deep copy into a new contiguous matrix
        Matrix real_result(rows, cols);
        for(size_t i=0; i<rows; ++i)
            for(size_t j=0; j<cols; ++j)
                real_result(i,j) = (*this)(i,j);
        real_result.add(other);
        return real_result;
    }
    
    // if contiguous, standard copy is fine (but wait, copy ctor is shallow now!)
    // we need to explicitly deep copy for operators
    Matrix deep_copy(rows, cols);
    for(size_t i=0; i<rows; ++i)
         for(size_t j=0; j<cols; ++j)
             deep_copy(i,j) = (*this)(i,j);
             
    deep_copy.add(other);     
    return deep_copy;
}

Matrix Matrix::operator-(const Matrix& other) const {
    // deep copy
    Matrix deep_copy(rows, cols);
    for(size_t i=0; i<rows; ++i)
         for(size_t j=0; j<cols; ++j)
             deep_copy(i,j) = (*this)(i,j);

    deep_copy.subtract(other);
    return deep_copy;
}

Matrix Matrix::operator*(float scalar) const {
    Matrix deep_copy(rows, cols);
    for(size_t i=0; i<rows; ++i)
         for(size_t j=0; j<cols; ++j)
             deep_copy(i,j) = (*this)(i,j);
             
    deep_copy.scale(scalar);
    return deep_copy;
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
    if (is_contiguous() && other.is_contiguous()) {
        vDSP_dotpr(
            raw_data(), 1, 
            other.raw_data(), 1, 
            &result, 
            rows * cols
        );
    } else {
        // slow path
        for(size_t i=0; i<rows; ++i)
             for(size_t j=0; j<cols; ++j)
                 result += (*this)(i,j) * other(i,j);
    }

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
    // [modified]: ensure we work on a contiguous copy
    Matrix L(rows, cols);
    for(size_t i=0; i<rows; ++i)
        for(size_t j=0; j<cols; ++j)
            L(i,j) = (*this)(i,j);

    int n = (int)rows;
    int lda = n;
    int info = 0;
    
    // calling the lapack routine directly
    // "L" means fill the lower triangle. 
    // careful: lapack expects mutable pointers
    spotrf_("L", &n, L.raw_data(), &lda, &info);

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
    // [modified]: O(1) VIEW IMPLEMENTATION!
    // we just swap rows/cols and swap strides. no data copying.
    return Matrix(cols, rows, data, offset, stride_cols, stride_rows);
}

// LU decomposition using lapack sgetrf
// this is the bread and butter solver for general systems
Matrix Matrix::lu() const {
    if (rows != cols) {
        throw std::invalid_argument("lu decomp requires square matrix");
    }

    // [modified]: ensure contiguous copy
    Matrix Result(rows, cols);
    for(size_t i=0; i<rows; ++i)
        for(size_t j=0; j<cols; ++j)
            Result(i,j) = (*this)(i,j);
    
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
    sgetrf_(&n, &n, Result.raw_data(), &lda, ipiv.data(), &info);

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
    
    // [modified]: explicit deep copy for lapack input
    std::vector<float> a_copy(rows * cols);
    for(size_t i=0; i<rows; ++i)
         for(size_t j=0; j<cols; ++j)
             a_copy[j*rows + i] = (*this)(i,j); // ensure col-major packing
    
    sgesvd_(
        "A", "A", 
        &m, &n, 
        a_copy.data(), &lda, 
        S.raw_data(), 
        U.raw_data(), &ldu, 
        Vt.raw_data(), &ldvt, 
        work.data(), &lwork, 
        &info
    );

    if (info > 0) {
        throw std::runtime_error("svd failed to converge");
    }

    return {U, S, Vt};
}

// [NEW] Determinant
// Calculates "how much volume" the matrix represents. 0 means it's flat (singular).
float Matrix::determinant() const {
    if (rows != cols) throw std::invalid_argument("determinant requires square matrix");
    
    Matrix A_copy = this->clone();
    int n = (int)rows;
    int lda = n;
    std::vector<int> ipiv(n);
    int info = 0;
    
    // LU Decomp
    sgetrf_(&n, &n, A_copy.raw_data(), &lda, ipiv.data(), &info);
    
    if (info > 0) return 0.0f; // Singular

    // Det is product of diagonal of U * (-1)^swaps
    float det = 1.0f;
    for(size_t i=0; i<rows; ++i) {
        det *= A_copy(i, i);
        // check if swap happened (ipiv is 1-based in standard lapack, checking deviation)
        if (ipiv[i] != (int)(i + 1)) {
            det = -det;
        }
    }
    return det;
}

// [NEW] Inverse
// Undoing the matrix multiplication. A * A^-1 = I.
Matrix Matrix::inverse() const {
    if (rows != cols) throw std::invalid_argument("inverse requires square matrix");
    
    Matrix I = Matrix::identity(rows);
    
    // Solving A * X = I using LU
    Matrix A_copy = this->clone();
    Matrix X = I; // Result X starts as Identity
    
    int n = (int)rows;
    int nrhs = (int)rows;
    int lda = n;
    int ldb = n;
    std::vector<int> ipiv(n);
    int info = 0;
    
    // 1. Factorize A
    sgetrf_(&n, &n, A_copy.raw_data(), &lda, ipiv.data(), &info);
    if (info != 0) throw std::runtime_error("Matrix is singular, cannot invert (div by zero basically)");
    
    // 2. Solve for X
    sgetrs_("N", &n, &nrhs, A_copy.raw_data(), &lda, ipiv.data(), X.raw_data(), &ldb, &info);
    
    return X;
}

// Deep copy helper
Matrix Matrix::clone() const {
    Matrix copy(rows, cols); // allocates new clean storage
    
    if (is_contiguous()) {
        // [FIX]: Copy from raw_data() (start of view), not data->begin() (start of storage)
        // Also only copy rows*cols elements
        std::copy(raw_data(), raw_data() + (rows * cols), copy.data->begin());
    } else {
        // Fallback for strided views
        for(size_t i=0; i<rows; ++i)
            for(size_t j=0; j<cols; ++j)
                copy(i,j) = (*this)(i,j);
    }
    return copy;
}

// Solves A * X = B where A is Symmetric Positive Definite
// Wraps LAPACK 'sposv'
Matrix Matrix::solve_spd(const Matrix& B) const {
    if (rows != cols) throw std::invalid_argument("A must be square for solve");
    if (rows != B.rows) throw std::invalid_argument("Row mismatch between A and B");

    // sposv destroys A and B, so we must clone them
    Matrix A_copy = this->clone();
    Matrix X = B.clone(); // B gets overwritten with the solution X

    int n = (int)rows;
    int nrhs = (int)B.cols;
    int lda = n;
    int ldb = n; // B.rows
    int info = 0;

    // "L" = assume lower triangle is stored
    sposv_("L", &n, &nrhs, A_copy.raw_data(), &lda, X.raw_data(), &ldb, &info);

    if (info != 0) {
        throw std::runtime_error("solve_spd failed (matrix might not be positive definite)");
    }

    return X;
}

Matrix Matrix::row(size_t i) const {
    if (i >= rows) throw std::out_of_range("row index out of bounds");
    // New view starts at offset + i * stride_rows
    // It has 1 row, 'cols' columns
    // Stride for cols remains same, stride for rows is irrelevant (since only 1 row)
    return Matrix(1, cols, data, offset + i * stride_rows, stride_rows, stride_cols);
}

Matrix Matrix::col(size_t j) const {
    if (j >= cols) throw std::out_of_range("col index out of bounds");
    // New view starts at offset + j * stride_cols
    // It has 'rows' rows, 1 column
    // Stride for rows remains same
    return Matrix(rows, 1, data, offset + j * stride_cols, stride_rows, stride_cols);
}

Matrix Matrix::apply(std::function<float(float)> func) const {
    Matrix result(rows, cols);
    // If contiguous, we can loop linearly (optimization)
    if (is_contiguous()) {
        for(size_t i=0; i<data->size(); ++i) {
             (*result.data)[i] = func((*data)[i + offset]);
        }
    } else {
        // Safe fallback for views
        for(size_t i=0; i<rows; ++i) {
            for(size_t j=0; j<cols; ++j) {
                result(i,j) = func((*this)(i,j));
            }
        }
    }
    return result;
}


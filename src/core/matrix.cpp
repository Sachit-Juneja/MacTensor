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
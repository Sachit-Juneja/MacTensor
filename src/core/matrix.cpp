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
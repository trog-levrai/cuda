#pragma once

# include <stdexcept>
# include <cuda_runtime.h>
# include <cuda.h>
# include "cublas_v2.h"
# include "matrix_helper.cuh"

class CudaMatrix {
  public:
    //Destructor of the class
    ~CudaMatrix();

    //Constructor that copies host matrix to device
    CudaMatrix(cublasHandle_t handle, size_t M, size_t N, const float* a_h);

    //Constructor that creates device matrix
    CudaMatrix(cublasHandle_t handle, size_t M, size_t N);

    //Copy constructor
    CudaMatrix(const CudaMatrix&);

    //Dot product of matrix
    CudaMatrix operator*(const CudaMatrix&);

    //Multiplication with a scalar
    CudaMatrix operator*(float);

    //Cell-wise multiplication
    CudaMatrix operator%(const CudaMatrix&);

    //Cell-wise substraction
    CudaMatrix operator-(const CudaMatrix&);

    //Cell-wise addition
    CudaMatrix operator+(const CudaMatrix&);

    //Substraction of a scalar
    CudaMatrix operator-(float);

    //Addition of a scalar
    CudaMatrix operator+(float);

    //Randomly fills the matrix
    void randomize();

    //Returns the transpose of the current matrix
    CudaMatrix t() const;

    //Transform and return a matrix by a func
    CudaMatrix& transform(std::function<float (float)> f);
    
    //Reshape
    CudaMatrix& reshape(size_t M, size_t N);

  private:
    float* a_d_;
    cublasHandle_t handle_;
  public:
    size_t M_;
    size_t N_;
};

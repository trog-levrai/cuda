#pragma once

# include <stdexcept>
# include <cuda_runtime.h>
# include <cuda.h>
# include <unistd.h>
# include <thrust/device_vector.h>
# include <thrust/reduce.h>
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
    CudaMatrix operator*(const CudaMatrix&) const;

    //Multiplication with a scalar
    CudaMatrix operator*(float) const;

    //Cell-wise multiplication
    CudaMatrix operator%(const CudaMatrix&) const;

    //Cell-wise substraction
    CudaMatrix operator-(const CudaMatrix&) const;

    //Cell-wise addition
    CudaMatrix operator+(const CudaMatrix&) const;

    //Substraction of a scalar
    CudaMatrix operator-(float) const;

    //Addition of a scalar
    CudaMatrix operator+(float) const;

    //Randomly fills the matrix
    void randomize();

    //Returns the transpose of the current matrix
    CudaMatrix t() const;

    //Transform and return a matrix by a func
    CudaMatrix transform(float (*f)(float));

    //Transform and return a matrix by a func
    CudaMatrix relu();

    //Transform and return a matrix by a func
    CudaMatrix d_relu();

    //Reshape
    CudaMatrix reshape(size_t M, size_t N);

    //set mat
    void setMat(float* arr) { a_d_ = arr; };

    //Sums up all the elements of the matrix
    float accu() const;

    //Returns the column at the position indicated in the parameter
    CudaMatrix rows(size_t, size_t) const;

    //Returns the column at indeces
    CudaMatrix rows(std::vector<size_t>&) const;

    //Insert a column of 1
    void addBias(); 

  private:
    float* a_d_;
    cublasHandle_t handle_;
  public:
    size_t M_;
    size_t N_;
};

CudaMatrix ones(size_t M, size_t N, cublasHandle_t handle);

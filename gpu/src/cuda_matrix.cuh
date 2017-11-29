#pragma once

# include <stdexcept>
# include <cuda_runtime.h>
# include <cuda.h>
# include <cuda_fp16.h>
# include <unistd.h>
# include <thrust/device_vector.h>
# include <thrust/execution_policy.h>
# include <thrust/reduce.h>
# include "cublas_v2.h"
# include "matrix_helper.cuh"

class CudaMatrix {
  public:
    //Destructor of the class
    ~CudaMatrix();

    //Constructor that copies host matrix to device
    CudaMatrix(cublasHandle_t handle, size_t M, size_t N, const float* a_h, bool);

    //Constructor that creates device matrix
    CudaMatrix(cublasHandle_t handle, size_t M, size_t N, bool);

    //Copy constructor
    CudaMatrix(const CudaMatrix&);

    //Dot product of matrix
    CudaMatrix operator*(const CudaMatrix&) const;

    //Doc product without realloc
    CudaMatrix mult_buff(const CudaMatrix& m, CudaMatrix& o) const;

    //Dot product of matrix
    CudaMatrix dot(const CudaMatrix&, float al) const;

    //Assignement operator
    CudaMatrix operator=(const CudaMatrix&);

    //Assignment without realloc
    CudaMatrix operator+=(const CudaMatrix& m);

    //Multiplication with a scalar
    CudaMatrix operator*(float) const;

    //Cell-wise multiplication
    CudaMatrix operator%(const CudaMatrix&) const;

    //Cell-wise mult
    CudaMatrix operator%=(const CudaMatrix& m);

    //Cell-wise substraction
    CudaMatrix operator-(const CudaMatrix&) const;

    //Cell-wise sub in place
    CudaMatrix operator-=(const CudaMatrix& m);

    //Cell-wise addition
    CudaMatrix operator+(const CudaMatrix&) const;

    //Substraction of a scalar
    CudaMatrix operator-(float) const;

    //Addition of a scalar
    CudaMatrix operator+(float) const;

    //Addition of a scalar inplace
    CudaMatrix operator+=(float);

    //Randomly fills the matrix
    void randomize();

    // Print
    void print() const;

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
    void setMat(float* arr) { a_d_ = std::shared_ptr<float>(arr, cudaFree); };

    //get mat
    std::shared_ptr<float> getMat() const { return f_d_; };

    //Sums up all the elements of the matrix
    float accu() const;

    //Returns the column at the position indicated in the parameter
    CudaMatrix rows(size_t, size_t) const;

    //Returns the column at indeces
    CudaMatrix rows(std::vector<size_t>&) const;

    //Insert a column of 1
    CudaMatrix addBias();

    //Returns the shape of the matrix as a pair
    std::pair<size_t, size_t> shape() const;

    //Prints the shape of th matrix
    void print_shape(std::string) const;

    //Return the float16 version of the matrix
    CudaMatrix getHalf() const;

    //Return the float32 version of the matrix
    CudaMatrix getSingle() const;

    //Allocates memory for this matrix
    void alloc();

  private:
    std::shared_ptr<float> f_d_ = NULL;
    std::shared_ptr<void> a_d_ = NULL;
    cublasHandle_t handle_;
    bool half_;
  public:
    size_t M_;
    size_t N_;
};

CudaMatrix ones(size_t M, size_t N, cublasHandle_t handle);

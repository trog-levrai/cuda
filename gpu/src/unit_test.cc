# include <iostream>

# include "cuda_matrix.cuh"
# include "model.hh"

int main() {
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;

  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf ("CUBLAS initialization failed\n");
    return EXIT_FAILURE;
  }

  float X[] = {1., 2., 3.,\
               4., 5., 6.,\
               7., 8., 9.};
  mat X_(handle, 3, 3, X);
  //Transpose
  //mat y = X_.t();

  //Addition of scalar
  //mat y = X_ + 1.;

  //Substraction of scalar
  //mat y = X_ - 1.;

  //Multiplication by scalar
  //mat y = X_ * 2.;

  //Dot product
  //mat y = X_ * X_;

  //Element-wise multiplication
  //mat y = X_ % X_;

  //Element-wise substraction
  //mat y = X_ - X_;

  //Element-wise addition
  mat y = X_ + X_;

  //Ones
  //mat y = ones(4, 3, handle);

  y.print();

  return 0;
}

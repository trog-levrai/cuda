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
               7., 8., 9.,\
              10., 11., 12.,\
              13., 14., 15.};
  float Y[] = {3.,\
               6.,\
               9.,\
              12.,\
              15.};
  mat X_(handle, 5, 3, X);
  mat Y_(handle, 5, 1, Y);
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
  //mat y = X_ + X_;

  //Ones
  //mat y = ones(4, 3, handle);

  //Random matrix
  //mat y = ones(4, 3, handle);
  //y.randomize();

  //Reshape matrix
  //mat y = ones(4, 3, handle);
  //y = y.reshape(3, 4);
  //y.randomize();

  //Accu
  //std::cout << X_.accu() << std::endl;

  //Rows interval
  //mat y = X_.rows(1, 3);

  //Rows indexes
  //auto aux = std::vector<size_t>();
  //aux.push_back(0);
  //aux.push_back(2);
  //aux.push_back(4);
  //mat y = X_.rows(aux);

  //Add Bias
  //mat y = X_;
  //y.addBias();

  //Relu
  //mat y = (X_ - 3).relu();

  //DRelu
  //mat y = (X_ - 3).d_relu();

  //Matrix addition with vector
  mat y = X_ - Y_;

  y.print();

  return 0;
}

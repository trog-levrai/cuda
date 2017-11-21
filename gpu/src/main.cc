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

  float X[] = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
  mat X_(handle, 3, 3, X);

  /*  float X2[] = {-1, 0, 0.5};
  mat X2_(handle, 1, 3, X2);

  X_ = X_ - 0.5;*/
  std::cout << X_.accu() << std::endl;

  return 0;
}

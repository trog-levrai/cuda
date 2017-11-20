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

//  Model m(handle);
//  m.add(1, 2);

  float X[] = {-1.5, 0, 1.5};
  mat X_(handle, 1, 3, X);
  X_ = X_.d_relu();
  X_.print();
//  Y_.print();
  return 0;
}

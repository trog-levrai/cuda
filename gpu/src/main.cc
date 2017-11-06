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

  Model m(handle);
  m.add(1, 2);

  return 0;
}

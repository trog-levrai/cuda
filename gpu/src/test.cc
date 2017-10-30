#include <malloc.h>
#include <iostream>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cuda_matrix.hh"

int main() {
  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "CUBLAS initialization failed\n";
    return 1;
  }
  float* foo = (float*)malloc(sizeof(float) * 4);
  float* bar = (float*)malloc(sizeof(float) * 4);
  for (size_t i = 0; i < 4; ++i) {
	  foo[i] = i;
	  bar[i] = 4 - i;
  }

  CudaMatrix a = CudaMatrix(handle, 2, 2, foo);
  CudaMatrix b = CudaMatrix(handle, 2, 2, bar);

  a * b;
}

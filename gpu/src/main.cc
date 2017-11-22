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

  float X[] = {0., 0., 1., 1.,\
               0., 1., 0., 1.};
  float Y[] = {0., 0., 0., 1.};
  mat y(handle, 1, 4, Y);
  mat X_(handle, 2, 4, X);
  X_ = X_.t();
  y = y.t();
  Model M(handle);
  //M.add(3, 2, "relu");
  M.add(1, 2, "relu");

  M.train(X_, y, 100, .1);
  mat out = M.forward(X_);

  M.forward(X_).print();

  /*  float X2[] = {-1, 0, 0.5};
  mat X2_(handle, 1, 3, X2);

  X_ = X_ - 0.5;*/

  return 0;
}

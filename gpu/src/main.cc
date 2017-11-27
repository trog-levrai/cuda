# include <iostream>

# include "cuda_matrix.cuh"
# include "model.hh"
# include "mnist.hh"

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
  float Y[] = {1., -1., -1., 1.};

  {
    /*
    mat y(handle, 1, 4, Y);
    mat X_(handle, 2, 4, X);

    X_ = X_.t();
    y = y.t();

    Model M(handle);
    M.add(3, 2, "relu");
    M.add(1, "relu");
    M.compile();

    M.train(X_, y, 10000, 0.1);

    M.forward(X_).print();
    */
    std::string filename = "../data/train-images-idx3-ubyte";
    Mnist::read_Mnist(filename, handle);
  }

  cublasDestroy(handle);

  cudaDeviceReset();

  return 0;
}

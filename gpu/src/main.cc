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
    auto trainData = Mnist::read_Mnist(filename, handle);
    std::string filenameLabels = "../data/train-labels-idx1-ubyte";
    auto trainLabels = Mnist::read_Mnist_Label(filenameLabels, handle, 60000);
    //std::string filenameTest = "../data/train-images-idx3-ubyte";
    //auto testData = Mnist::read_Mnist(filenameTest, handle);
    //std::string filenameLabelsTest = "../data/train-labels-idx1-ubyte";
    //auto testLabels = Mnist::read_Mnist_Label(filenameLabelsTest, handle, 10000);

    Model M(handle);
    M.add(40, 784);
    M.add(10);

    M.compile();

    M.train(trainData, trainLabels, 100, 0.1);
  }

  cublasDestroy(handle);

  cudaDeviceReset();

  return 0;
}

#pragma once

# include <math.h>
# include <fstream>

# include "cuda_matrix.cuh"

#define MNIST_IMG_SIZE 784
#define NB_CLASS 10

typedef CudaMatrix mat;

class Mnist {
  public:
    static int ReverseInt (int);
    static mat read_Mnist(std::string, cublasHandle_t);
    static mat read_Mnist_Label(std::string, cublasHandle_t, size_t);
};

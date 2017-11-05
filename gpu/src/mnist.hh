#pragma once

# include <math.h>
# include <iostream>

# include "cuda_matrix.hh"

typedef CudaMatrix mat;

class Mnist {
  public:
    static int ReverseInt (int);
    static void read_Mnist(std::string, std::vector<mat>&);
    static void read_Mnist_Label(std::string, arma::colvec&);
};

#pragma once

# include <armadillo>
# include <math.h>
# include <iostream>

typedef arma::Mat<float> mat;

class Mnist {
  public:
    static int ReverseInt (int);
    static void read_Mnist(std::string, std::vector<mat>&);
    static void read_Mnist_Label(std::string, arma::colvec&);
};

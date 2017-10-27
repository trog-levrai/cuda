#pragma once

# include <armadillo>
# include <math.h>
# include <iostream>

class Mnist {
  public:
    static int ReverseInt (int);
    static void read_Mnist(std::string, std::vector<arma::mat>&);
    static void read_Mnist_Label(std::string, arma::colvec&);
};

# include <iostream>
# include <armadillo>

# include "model.hh"

int main() {
  Model M;
  M.add(1, 2);
  M.add(1);

  arma::Mat<float> X(4, 2);
  X = {{0, 0},
       {0, 1},
       {1, 0},
       {1, 1}};

  arma::Mat<float> y(1, 4);
  y = {{0, 1, 1, 0}};
  y = y.t();

  std::cout << M.forward(X) << std::endl;
  std::cout << y << std::endl;

  return 0;
}

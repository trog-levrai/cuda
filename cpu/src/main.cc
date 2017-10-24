# include <iostream>
# include <armadillo>

# include "model.hh"

int main() {
  Model M;
  M.add(1, 2);
  M.add(10);
  M.add(2);

  arma::Mat<float> A(1, 2);
  A = {{0, 0}};

  std::cout << M.forward(A) << std::endl;
}

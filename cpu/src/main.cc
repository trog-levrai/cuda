# include <iostream>
# include <armadillo>

# include "model.hh"

int main() {
  Model M;
  M.add(40, 2, "tan_h");
  M.add_max_POOL();
  M.add(1, "tan_h");

  arma::Mat<float> X(4, 2);
  X = {{0, 0},
       {0, 1},
       {1, 0},
       {1, 1}};

  arma::Mat<float> y(1, 4);
  y = {{1, -1, -1, 1}};
  y = y.t();

  M.train(X, y, 40000, 0.01);

  std::cout << M.forward(X) << std::endl;
  std::cout << y << std::endl;

  return 0;
}

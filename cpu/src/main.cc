# include <iostream>
# include <armadillo>

# include "model.hh"

int main() {

  Model M;
  M.add(3, 2);
  M.add(2);

  arma::Mat<float> X(4, 2);
  X = {{0, 0},
       {0, 1},
       {1, 0},
       {1, 1}};

  arma::Mat<float> y(2, 4);
  y = {{-1, 1, 1, -1},
       {1, -1, -1, 1}};
  y = y.t();

  M.train(X, y, 10);

  std::cout << M.forward(X) << std::endl;
  std::cout << y << std::endl;
  
/*
  Model M;
  M.add(5, 2);
  M.add(1);

  arma::Mat<float> X(4, 2);
  X = {{0, 0},
       {0, 1},
       {1, 0},
       {1, 1}};

  arma::Mat<float> y(2, 4);
  y = {{1, -1, -1, 1}};
  y = y.t();

  M.train(X, y, 10);

  std::cout << M.forward(X) << std::endl;
  std::cout << y << std::endl;
*/

  return 0;
}

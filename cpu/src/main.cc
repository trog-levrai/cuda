# include <iostream>
# include <armadillo>

# include "model.hh"

int main() {

  auto A = arma::Mat<float>(3, 3);
  A = {{1, 2, 3},
       {1, 2, 3},
       {1, 2, 3}};

  std::cout << "A:\n" << A << "\n";
  std::cout << A * A.t() << "\n";
}

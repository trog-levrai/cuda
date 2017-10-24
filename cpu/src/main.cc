# include <iostream>
# include <armadillo>

# include "model.hh"

int main() {

  auto A = arma::Mat<float>(3, 3);
  A = {{1, 2, 3},
       {1, 2, 3},
       {1, 2, 3}};

  auto m = Model();

  std::cout << "A:\n" << A << "\n";
  std::cout << A * A.t() << "\n";
  std::cout << m.dsigmoid_mat_(A) << std::endl;
  std::cout << m.sigmoid_mat_(A) << std::endl;
}

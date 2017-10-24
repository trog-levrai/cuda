#include <iostream>
#include <armadillo>

int main() {
  arma::arma_rng::set_seed_random();
  arma::Mat<double> A = arma::randu(4,4);
  std::cout << "A:\n" << A << "\n";
  std::cout << A * A.t() << "\n";
}

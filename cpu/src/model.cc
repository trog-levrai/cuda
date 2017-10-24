# include "model.hh"

# include <cmath>

void Model::init_W(size_t m, size_t n) {
  arma::mat M(m, n, fill::randu);
  float r = 4.0 * sqrt(6.0 / (m + n))
  M *= r;
  this->W.emplace_back(M);
}

arma::Mat<float> Model::sigmoid_mat_(arma::Mat<float>& matrix) {
  return matrix.transform( [](double x) { return 1.0 / (1.0 + std::exp(-x)); } );
}

arma::Mat<float> Model::dsigmoid_mat_(arma::Mat<float>& matrix) {
  return matrix.transform( [](double x) { return x * (1.0 - x); } );
}

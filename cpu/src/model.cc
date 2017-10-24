# include "model.hh"

# include <cmath>

void Model::init_W(size_t m, size_t n) {
  arma::mat M(m, n, fill::randu);
  float r = 4.0 * sqrt(6.0 / (m + n))
  M *= r;
  this->W.emplace_back(M);
}

float Model::sigmoid_(float x) {
  return 1.0 / (1.0 + std::exp(-x));
}

float Model::dsigmoid_(float x) {
  return x * (1.0 - x);
}

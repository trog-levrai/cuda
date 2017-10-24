# include "model.hh"

# include <cmath>

void Model::init_W(size_t m, size_t n)
{
  arma::mat M(m, n, fill::randu);
  float r = 4.0 * sqrt(6.0 / (m + n))
  M *= r;
  this->W.emplace_back(M);
}

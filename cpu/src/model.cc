# include "model.hh"

# include <cmath>
# include <stdexcept>

void Model::init_W(size_t m, size_t n) {
  arma::Mat<float> M;
  M.randu(m, n);
  float r = 4.0 * sqrt(6.0 / (m + n));
  M *= r;
  this->W.emplace_back(M);
}

float Model::sigmoid_(float x) {
  return 1.0 / (1.0 + std::exp(-x));
}

float Model::dsigmoid_(float x) {
  return x * (1.0 - x);
}

void Model::add(size_t output_units, size_t input_units) {
  if (!this->W.empty())
    throw std::runtime_error("An input layer has already been add");

  init_W(input_units + 1, output_units);
}

void Model::add(size_t output_units) {
  if (this->W.empty())
    throw std::runtime_error("The model has no input layer");

  size_t input_units = this->W.back().n_cols;
  init_W(input_units + 1, output_units);
}

arma::Mat<float>& Model::forward(arma::Mat<float>& X) {
  arma::Mat<float> X_(X);
  arma::Mat<float> tmp = ones<arma::Mat<float>>(X_.n_rows, 1);
  X_.insert_cols(X_.n_cols, tmp);

  auto o = X_ * this->W[0];
  return this->sigmoid_(o);
}

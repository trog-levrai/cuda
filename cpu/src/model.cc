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

arma::Mat<float> Model::sigmoid_mat_(arma::Mat<float>& matrix) {
  return matrix.transform( [](double x) { return 1.0 / (1.0 + std::exp(-x)); } );
}

arma::Mat<float> Model::dsigmoid_mat_(arma::Mat<float>& matrix) {
  return matrix.transform( [](double x) { return x * (1.0 - x); } );
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

arma::Mat<float> Model::forward(const arma::Mat<float>& X) {
  arma::Mat<float> X_c(X);
  for (auto& W_ : this->W)
  {
    arma::Mat<float> X_(X_c);
    arma::Mat<float> tmp = arma::ones<arma::Mat<float>>(X_.n_rows, 1);
    X_.insert_cols(X_.n_cols, tmp);

    arma::Mat<float> o = X_ * W_;
    X_c = this->sigmoid_mat_(o);
  }
  return X_c;
}

arma::Mat<float> Model::forward_keep(const arma::Mat<float>& X) {
  arma::Mat<float> X_c(X);
  for (auto& W_ : this->W)
  {
    arma::Mat<float> X_(X_c);
    arma::Mat<float> tmp = arma::ones<arma::Mat<float>>(X_.n_rows, 1);
    X_.insert_cols(X_.n_cols, tmp);

    arma::Mat<float> o = X_ * W_;
    H.push_back(arma::Mat<float>(o));
    X_c = this->sigmoid_mat_(o);
    C.push_back(X_c);
  }
  return X_c;
}

std::vector<arma::Mat<float>> Model::get_err(const arma::Mat<float> truth) {
  auto err0 = (truth - C.back()) * dsigmoid_mat_(H.back());
  auto err_vec = std::vector<arma::Mat<float>>();
  err_vec.push_back(err0);
  for (int i = W.size() - 2; i > 0; --i) {
    auto err = dsigmoid_mat_(H[i]) * arma::sum((W[i] * err_vec.back()), 1);
    err_vec.push_back(err);
  }

  std::reverse(err_vec.begin(), err_vec.end());
  return err_vec;
}

/*
 *void Model::back_propagate(float lambda, const arma::Mat<float> truth) {
 *  auto err = get_err(truth);
 *}
 */

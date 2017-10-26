# include "model.hh"

# include <cmath>
# include <stdexcept>
# include <assert.h>

void Model::init_W(size_t m, size_t n) {
  arma::Mat<float> M;
  M.randu(m, n);
  M -= 0.5;
  float r = 4.0 * sqrt(6.0 / (m + n));
  M *= r;
  this->W.emplace_back(M);
}

arma::Mat<float> Model::sigmoid_mat_(arma::Mat<float>& matrix) {
  return matrix.transform( [](double x) { return (std::exp(2 * x) - 1) / (std::exp(2 * x) + 1); } );
}

arma::Mat<float> Model::dsigmoid_mat_(arma::Mat<float>& matrix) {
  return matrix.transform( [](double x) {
      float h = (std::exp(2 * x) - 1) / (std::exp(2 * x) + 1);
      return 1 - h * h; } );
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
  this->H.clear();
  this->C.clear();

  arma::Mat<float> X_c(X);
  arma::Mat<float> X_(X_c);
  arma::Mat<float> tmp = arma::ones<arma::Mat<float>>(X_.n_rows, 1);
  X_.insert_cols(X_.n_cols, tmp);

  for (auto& W_ : this->W)
  {
    arma::Mat<float> X_(X_c);
    arma::Mat<float> tmp = arma::ones<arma::Mat<float>>(X_.n_rows, 1);
    X_.insert_cols(X_.n_cols, tmp);
    this->C.push_back(arma::Mat<float>(X_));

    arma::Mat<float> o(X_ * W_);

    this->H.push_back(arma::Mat<float>(o));

    X_c = this->sigmoid_mat_(o);

  }
  this->C.push_back(X_c);
  return X_c;
}

std::vector<arma::Mat<float>> Model::get_err(const arma::Mat<float> truth) {
  arma::Mat<float> cp(H.back());
  arma::Mat<float> err0 = (truth - C.back()) % dsigmoid_mat_(cp);

  auto err_vec = std::vector<arma::Mat<float>>();
  err_vec.push_back(err0);

  for (int i = W.size() - 2; i >= 0; --i) {
    arma::Mat<float> tmp = this->W[i + 1] * err_vec.back().t();
    arma::Mat<float> aux = tmp.rows(0, tmp.n_rows - 2); //TOOO

    arma::Mat<float> cp2(H[i]);
    arma::Mat<float> err = aux.t() % dsigmoid_mat_(cp2);

    err_vec.push_back(err);
  }

  std::reverse(err_vec.begin(), err_vec.end());
  return err_vec;
}

void Model::back_propagate(float lambda, const arma::Mat<float> truth) {
  auto err = get_err(truth);

  for (size_t i = 0; i < this->W.size(); ++i) {
    arma::Mat<float> tmp = (lambda * err[i].t() * this->C[i]);
    this->W[i] += tmp.t();
  }
}

float Model::loss(const arma::Mat<float>& X, const arma::Mat<float>& y) {
  arma::Mat<float> out = this->forward(X);
  out = (out - y);
  out = out % out;
  return arma::accu(out) / y.n_rows;
}

void Model::train(arma::Mat<float>& X, arma::Mat<float>& y, size_t nb_epoch, float lr) {
  for (size_t i = 0; i < nb_epoch; i++)
  {
    auto shuffle = std::vector<size_t>();
    for (size_t i = 0; i < X.n_rows; ++i) {
      shuffle.push_back(i);
    }
    std::random_shuffle(shuffle.begin(), shuffle.end());

    std::cout << "============ EPOCH " << i << "\n";

    for (size_t j = 0; j < shuffle.size(); ++j)
    {
      this->forward_keep(X.row(shuffle[j]));
      this->back_propagate(lr, y.row(shuffle[j]));
    }

    std::cout << "Train loss: " << this->loss(X, y) << std::endl;
    std::cout << std::endl;
  }
}

void Model::train(arma::Mat<float>& X, arma::Mat<float>& y, size_t nb_epoch) {
  this->train(X, y, nb_epoch, 0.1);
}

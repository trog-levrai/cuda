# include "model.hh"

# include <cmath>
# include <stdexcept>
# include <assert.h>

void Model::init_W(size_t m, size_t n) {
  mat M;
  M.randu(m, n);
  M -= 0.5;
  float r = 4.0 * sqrt(6.0 / (m + n));
  M *= r;
  this->W.emplace_back(M);
}

mat Model::activate(mat& matrix, const std::string func) {
  return matrix.transform( map_func[func]->f() );
}

mat Model::d_activate(mat& matrix, const std::string func) {
  return matrix.transform( map_func[func]->d_f() );
}

void Model::add(size_t output_units, size_t input_units) {
  this->add(output_units, input_units, "tan_h");
}


void Model::add(size_t output_units) {
  this->add(output_units, "tan_h");
}

void Model::add(size_t output_units, size_t input_units, std::string activ) {
  if (!this->W.empty())
    throw std::runtime_error("An input layer has already been add");

  init_W(input_units + 1, output_units);

  this->activate_vec.push_back(activ);
}

void Model::add(size_t output_units, std::string activ) {
  if (this->W.empty())
    throw std::runtime_error("The model has no input layer");

  size_t input_units = this->W.back().n_cols;
  init_W(input_units + 1, output_units);

  this->activate_vec.push_back("tan_h");
}

const mat Model::forward(const mat& X) {
  mat X_c(X);
  size_t i = 0;
  for (auto& W_ : this->W)
  {
    mat X_(X_c);
    mat tmp = arma::ones<mat>(X_.n_rows, 1);
    X_.insert_cols(X_.n_cols, tmp);

    mat o = X_ * W_;
    X_c = this->activate(o, this->activate_vec[i]);
    ++i;
  }
  return X_c;
}

mat Model::forward_keep(const mat& X) {
  this->H.clear();
  this->C.clear();

  mat X_c(X);
  mat X_(X_c);
  mat tmp = arma::ones<mat>(X_.n_rows, 1);
  X_.insert_cols(X_.n_cols, tmp);

  size_t i = 0;
  for (auto& W_ : this->W)
  {
    mat X_(X_c);
    mat tmp = arma::ones<mat>(X_.n_rows, 1);
    X_.insert_cols(X_.n_cols, tmp);
    this->C.push_back(mat(X_));

    mat o(X_ * W_);

    this->H.push_back(mat(o));

    X_c = this->activate(o, this->activate_vec[i]);
    ++i;
  }
  this->C.push_back(X_c);
  return X_c;
}

std::vector<mat> Model::get_err(const mat truth) {
  mat cp(H.back());
  mat err0 = (truth - C.back()) % this->d_activate(cp, this->activate_vec.back());

  auto err_vec = std::vector<mat>();
  err_vec.push_back(err0);

  for (int i = W.size() - 2; i >= 0; --i) {
    mat tmp = this->W[i + 1] * err_vec.back().t();
    mat aux = tmp.rows(0, tmp.n_rows - 2);

    mat cp2(H[i]);
    mat err = aux.t() % this->d_activate(cp2, this->activate_vec[i]);

    err_vec.push_back(err);
  }

  std::reverse(err_vec.begin(), err_vec.end());
  return err_vec;
}

void Model::back_propagate(float lambda, const mat truth) {
  auto err = get_err(truth);

  for (size_t i = 0; i < this->W.size(); ++i) {
    mat tmp = (lambda * err[i].t() * this->C[i]);
    this->W[i] += tmp.t();
  }
}

const float Model::loss(const mat& X, const mat& y) {
  mat out = this->forward(X);
  out = (out - y);
  out = out % out;
  return arma::accu(out) / y.n_rows;
}

void Model::train(const mat& X, const mat& y, size_t nb_epoch, float lr) {
  if (this->W.empty())
    throw std::runtime_error("An model has no input layer");

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

void Model::train(const mat& X, const mat& y, size_t nb_epoch) {
  this->train(X, y, nb_epoch, 0.1);
}

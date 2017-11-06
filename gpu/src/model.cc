# include "model.hh"

# include <cmath>
# include <stdexcept>
# include <assert.h>

void Model::init_W(size_t m, size_t n) {
  mat M;
  M.randomize();
  M -= 0.5;
  float r = 4.0 * sqrt(6.0 / (m + n));
  M *= r;
  this->W.emplace_back(M);
}

const mat Model::activate(mat& matrix, const std::string func) {
  return matrix.transform(map_func.at(func)->f());
}

const mat Model::d_activate(mat& matrix, const std::string func) {
  return matrix.transform(map_func.at(func)->d_f());
}

void Model::add(size_t output_units, size_t input_units) {
  this->add(output_units, input_units, "tan_h");
}


void Model::add(size_t output_units) {
  this->add(output_units, "tan_h");
}

void Model::add_max_POOL() {
  if (this->W.empty())
    throw std::runtime_error("The model has no input layer");

  if (this->W.back().n_cols % 4 != 0)
    throw std::runtime_error("The previus layer has to be a multiple of 4");

  this->type.push_back("pool");
  this->activate_vec.push_back("pool");
  this->init_W(1, this->W.back().n_cols / 4);
}

void Model::add(size_t output_units, size_t input_units, std::string activ) {
  if (!this->W.empty())
    throw std::runtime_error("An input layer has already been add");

  this->init_W(input_units + 1, output_units);

  this->activate_vec.push_back(activ);
  this->type.push_back("dense");
}

void Model::add(size_t output_units, std::string activ) {
  if (this->W.empty())
    throw std::runtime_error("The model has no input layer");

  size_t input_units = this->W.back().n_cols;

  init_W(input_units + 1, output_units);

  this->activate_vec.push_back("tan_h");
  this->type.push_back("dense");
}

const mat Model::forward(const mat& X) {
  mat X_c(X);
  size_t i = 0;
  for (auto& W_ : this->W) {
    if (this->type[i] == "dense") {
      mat X_(X_c);
      mat tmp = arma::ones<mat>(X_.n_rows, 1);

      X_.insert_cols(X_.n_cols, tmp);

      mat o = X_ * W_;
      X_c = this->activate(o, this->activate_vec[i]);
    }

    else if (this->type[i] == "pool") {
      size_t n_rows = X_c.n_rows;
      size_t n_cols = X_c.n_cols / 4;

      mat resh_x(X_c);
      resh_x.reshape(n_rows * n_cols, 4);
      X_c = mat(n_rows * n_cols, 1);

      for (size_t j = 0; j < X_c.n_rows; ++j) {
        float max = resh_x(j, 0);
        for (size_t k = 1; k < 4; ++k)
          max = max < resh_x(j, k) ? resh_x(j, k) : max;
        X_c(j, 0) = max;
      }
      X_c.reshape(n_rows, n_cols);
    }
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
  for (auto& W_ : this->W) {
    if (this->type[i] == "dense") {
      mat X_(X_c);
      mat tmp = arma::ones<mat>(X_.n_rows, 1);
      X_.insert_cols(X_.n_cols, tmp);
      this->C.push_back(mat(X_));

      mat o(X_ * W_);

      this->H.push_back(mat(o));

      X_c = this->activate(o, this->activate_vec[i]);
    }
    else if (this->type[i] == "pool") {
      this->C.push_back(mat(X_c));

      size_t n_rows = X_c.n_rows;
      size_t n_cols = X_c.n_cols / 4;

      mat resh_x(X_c);
      mat hh(X_c);
      hh.fill(0.0);

      resh_x.reshape(n_rows * n_cols, 4);
      X_c = mat(n_rows * n_cols, 1);

      for (size_t j = 0; j < X_c.n_rows; ++j) {
        float max = resh_x(j, 0);
        size_t id = 0;
        for (size_t k = 1; k < 4; ++k)
        {
          max = max < resh_x(j, k) ? resh_x(j, k) : max;
          id = max == resh_x(j, k) ? k : id;
        }
        X_c(j, 0) = max;
        hh(j * 4 + id) = 1; 
      }
      X_c.reshape(n_rows, n_cols);

      this->H.push_back(mat(hh));
    }
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
    if (this->type[i] == "dense") {
      mat tmp;
      mat aux;
      if (this->type[i + 1] == "dense")
      {
        tmp = this->W[i + 1] * err_vec.back().t();
        aux = tmp.rows(0, tmp.n_rows - 2);
      }
      else if (this->type[i + 1] == "pool")
        aux = err_vec.back().t();

      mat cp2(H[i]);
      mat err = aux.t() % this->d_activate(cp2, this->activate_vec[i]);

      err_vec.push_back(err);
    }
    else if (this->type[i] == "pool") {
      mat tmp = this->W[i + 1] * err_vec.back().t();
      mat aux = tmp.rows(0, tmp.n_rows - 2);
      aux = aux.t();
      mat aux_t(aux.n_rows, aux.n_cols * 4);

      for (size_t ii = 0; ii < aux_t.n_rows; ++ii)
        for (size_t jj = 0; jj < aux_t.n_cols; ++jj)
          aux_t(ii, jj) = aux(ii, jj / 4);

      mat err = aux_t % H[i];
      err_vec.push_back(err);
    }
  }

  std::reverse(err_vec.begin(), err_vec.end());
  return err_vec;
}

void Model::back_propagate(float lambda, const mat truth) {
  auto err = get_err(truth);

  for (size_t i = 0; i < this->W.size(); ++i) {
    if (this->type[i] != "pool")
    {
      mat tmp = (lambda * err[i].t() * this->C[i]);
      this->W[i] += tmp.t();
    }
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

# include "model.hh"

# include <cmath>
# include <stdexcept>

void Model::init_W(size_t m, size_t n) {
  arma::Mat<float> M;
  M.randu(m, n);
  //float r = 4.0 * sqrt(6.0 / (m + n));
  //M *= r;
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

    arma::Mat<float> o = X_ * W_;

    this->H.push_back(arma::Mat<float>(o));

    X_c = this->sigmoid_mat_(o);

  }
  this->C.push_back(X_c);
  return X_c;
}

std::vector<arma::Mat<float>> Model::get_err(const arma::Mat<float> truth) {
  arma::Mat<float> err0 = (truth - C.back()) % dsigmoid_mat_(H.back());

  auto err_vec = std::vector<arma::Mat<float>>();
  err_vec.push_back(err0);


  for (int i = W.size() - 2; i >= 0; --i) {
    /*std::cout << "W+1" << std::endl;
    std::cout << this->W[i+1] << std::endl;
    std::cout << "err" << std::endl;
    std::cout << err_vec.back().t() << std::endl;*/
    arma::Mat<float> tmp = this->W[i + 1] * err_vec.back().t();

    arma::Mat<float> tt = arma::sum(tmp, 0);

    arma::Mat<float> err = tt * dsigmoid_mat_(H[i]);

    err_vec.push_back(err);
  }

  std::reverse(err_vec.begin(), err_vec.end());
  return err_vec;
}

void Model::back_propagate(float lambda, const arma::Mat<float> truth) {
  auto err = get_err(truth);

  for (size_t i = 0; i < this->W.size(); ++i) {
    //std::cout << i << std::endl;
    //std::cout << W[i] << std::endl;
    arma::Mat<float> tmp = (lambda * err[i].t() * this->C[i]);
    std::cout << tmp << std::endl;
    this->W[i] += tmp.t();
    //std::cout << W[i] << std::endl;
  }
}

void Model::train(arma::Mat<float>& X, arma::Mat<float>& y, size_t nb_epoch) {
  for (size_t i = 0; i < nb_epoch; i++)
  {
    std::cout << "============ EPOCH " << i << "\n";
    // TODO SUFFLE DATA

    for (size_t j = 0; j < X.n_rows; j++)
    {
      //std::cout << j << "\n";
      //std::cout << X.row(j) << "\n";
      this->forward_keep(X.row(j));
      //std::cout << y.row(j) << "\n";
      this->back_propagate(1, y.row(j));
    }
    //std::cout << this->forward(X) << "\n";
  }
}

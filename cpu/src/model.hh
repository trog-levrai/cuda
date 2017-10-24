# pragma once

# include <math.h>
# include <vector>
# include <armadillo>

class Model {
  public:
    void add(size_t output_units);
    void add(size_t output_units, size_t input_units);
    arma::Mat<float>& predict(arma::Mat<float>& X);
    void train(arma::Mat<float>& X, arma::Mat<float>& y, size_t nb_epoch=10);
    arma::Mat<float> forward(const arma::Mat<float>& X);

  private:
    void init_W(size_t input, size_t output);
    arma::Mat<float> sigmoid_mat_(arma::Mat<float>&);
    arma::Mat<float> dsigmoid_mat_(arma::Mat<float>&);
    arma::Mat<float> forward_keep(const arma::Mat<float>& X);

    std::vector<arma::Mat<float>> W;
};

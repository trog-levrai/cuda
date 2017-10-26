# pragma once

# include <math.h>
# include <vector>
# include <armadillo>
# include <algorithm>
# include <map>
# include <string>

# include "activation_func.hh"

static std::map<std::string, activation_func*> map_func = {
  { "tan_h", new tan_h() },
  { "relu" , new relu{} }
};


class Model {

  public:
    void add(size_t output_units);
    void add(size_t output_units, size_t input_units);

    void add(size_t output_units, std::string activ);
    void add(size_t output_units, size_t input_units, std::string activ);

    arma::Mat<float>& predict(arma::Mat<float>& X);
    void train(const arma::Mat<float>& X, const arma::Mat<float>& y, size_t nb_epoch, float lr);
    void train(const arma::Mat<float>& X, const arma::Mat<float>& y, size_t nb_epoch);
    const arma::Mat<float> forward(const arma::Mat<float>& X);
    const float loss(const arma::Mat<float>& X, const arma::Mat<float>& y);

  private:
    void init_W(size_t input, size_t output);
    arma::Mat<float> activate(arma::Mat<float>&, const std::string func);
    arma::Mat<float> d_activate(arma::Mat<float>&, const std::string func);
    arma::Mat<float> forward_keep(const arma::Mat<float>& X);
    std::vector<arma::Mat<float>> get_err(const arma::Mat<float>);
    void back_propagate(float, const arma::Mat<float>);

    std::vector<std::string> activate_vec;

    std::vector<arma::Mat<float>> W;
    std::vector<arma::Mat<float>> H;
    std::vector<arma::Mat<float>> C;

};

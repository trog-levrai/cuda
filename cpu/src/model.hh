# pragma once

# include <math.h>
# include <vector>
# include <armadillo>
# include <algorithm>
# include <map>
# include <string>

# include "activation_func.hh"

typedef arma::Mat<float> mat;

static const std::map<const std::string, activation_func*> map_func = {
  { "tan_h", new tan_h() },
  { "relu" , new relu{} }
};

class Model {

  public:
    void add(size_t output_units);
    void add(size_t output_units, size_t input_units);

    void add(size_t output_units, std::string activ);
    void add(size_t output_units, size_t input_units, std::string activ);

    void train(const mat& X, const mat& y, size_t nb_epoch, float lr);
    void train(const mat& X, const mat& y, size_t nb_epoch);
    const mat forward(const mat& X);
    const float loss(const mat& X, const mat& y);

  private:
    void init_W(size_t input, size_t output);

    const mat activate(mat&, const std::string func);
    const mat d_activate(mat&, const std::string func);

    mat forward_keep(const mat& X);

    std::vector<mat> get_err(const mat);
    void back_propagate(float, const mat);

    std::vector<std::string> activate_vec;

    std::vector<mat> W;
    std::vector<mat> H;
    std::vector<mat> C;

};

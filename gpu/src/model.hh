# pragma once

# include <math.h>
# include <vector>
# include <algorithm>
# include <map>
# include <string>

//# include "activation_func.cuh"
# include "cuda_matrix.cuh"

typedef CudaMatrix mat;

/*
 *static const std::map<const std::string, activation_func*> map_func = {
 *  { "tan_h", new tan_h() },
 *  { "relu" , new relu{} }
 *};
 */

class Model {

  public:
    Model(cublasHandle_t handle)
    :handle_(handle), o_buff(handle_, 1, 1){}

    void add(size_t output_units);
    void add(size_t output_units, size_t input_units);

    void add(size_t output_units, std::string activ);
    void add(size_t output_units, size_t input_units, std::string activ);

    void add_max_POOL();

    void compile();

    void train(const mat& X, const mat& y, size_t nb_epoch, float lr);
    void train(const mat& X, const mat& y, size_t nb_epoch);
    const mat forward(const mat& X);
    const float loss(const mat& X, const mat& y);

  private:
    mat o_buff;

    void init_W(size_t input, size_t output);

    const mat activate(mat&, const std::string func);
    const mat d_activate(mat&, const std::string func);

    mat forward_keep(const mat& X);

    std::vector<mat> get_err(const mat);
    void back_propagate(float, const mat);

    std::vector<std::string> activate_vec;
    std::vector<std::string> type;

    std::vector<mat> W;
    std::vector<mat> H;
    std::vector<mat> C;
    cublasHandle_t handle_;

};

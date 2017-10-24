# pragma once

# include <vector>
#include <armadillo>

class Model {
  public:
    Model();
    void add(size_t output_units);
    void add(size_t output_units, size_t input_units);
    double* predict(double* X);
    void train(size_t nb_epoch=10);
  private:
    
    arma::Mat<float> normalize_(arma::Mat<float>&);
    arma::Mat<float>& sigmoid_mat_(arma::Mat<float>&);
    arma::Mat<float>& dsigmoid_mat_(arma::Mat<float>&);

    std::vector<double*> W;
};

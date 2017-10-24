# pragma once

# include <vector>

class Model {
  public:
    Model();
    void add(size_t output_units);
    void add(size_t output_units, size_t input_units);
    double* predict(double* X);
    void train(size_t nb_epoch=10);
  private:
    std::vector<double*> W;
};

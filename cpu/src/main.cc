#include <iostream>
#include <armadillo>

#include "model.hh"
#include "mnist.hh"

int main() {
  std::string filename = "../data/train-images-idx3-ubyte";

  std::vector<arma::mat> train_data;
  Mnist::read_Mnist(filename, train_data);
  
  std::string label_file = "../data/train-labels-idx1-ubyte";
  arma::colvec train_label = arma::zeros<arma::colvec>(train_data.size());
  Mnist::read_Mnist_Label(label_file, train_label);
  
  Model M;
  M.add(40, 2, "tan_h");
  M.add_max_POOL();
  M.add(1, "tan_h");

  arma::Mat<float> X(4, 2);
  X = {{0, 0},
       {0, 1},
       {1, 0},
       {1, 1}};

  arma::Mat<float> y(1, 4);
  y = {{1, -1, -1, 1}};
  y = y.t();

  M.train(X, y, 40000, 0.01);

  std::cout << M.forward(X) << std::endl;
  std::cout << y << std::endl;

  return 0;
}

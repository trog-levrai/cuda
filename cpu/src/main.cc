#include <iostream>
#include <armadillo>

#include "model.hh"
#include "mnist.hh"

int main() {
  std::string filename = "../data/train-images-idx3-ubyte";

  std::vector<mat> train_data;
  Mnist::read_Mnist(filename, train_data);
  
  std::string label_file = "../data/train-labels-idx1-ubyte";
  arma::colvec train_label = arma::zeros<arma::colvec>(train_data.size());
  Mnist::read_Mnist_Label(label_file, train_label);

  mat X(train_data.size(), 784);
  for (size_t i = 0; i < train_data.size(); ++i)
    for (size_t j = 0; j < 784; ++j)
      X(i, j) = train_data[i](j);

  mat y(train_data.size(), 10);
  y.fill(0.0);
  for (size_t i = 0; i < train_data.size(); ++i)
    y(i, train_label(i)) = 1.0;

  Model M;
  M.add(40, 784, "relu");
  M.add(10, "relu");

  M.train(X, y, 20, 0.01);
  mat out = M.forward(X);

  mat t(train_data.size(), 1);
  for (size_t i = 0; i < train_data.size(); ++i)
  {
    size_t id = 0;
    float max = out(i, 0);
    for (size_t j = 1; j < 10; ++j)
    {
      max = max < out(i, j) ? out(i, j) : max;
      id = max == out(i, j) ? j : id;
    }
    t(i, 0) = id;
  }

  float acc = 0;
  for (size_t i = 0; i < train_data.size(); ++i)
    acc = t(i, 0) == train_label(i) ? acc + 1 : acc;
  
  std::cout << "Accuracy= " << acc / train_data.size() << "\n";

  return 0;
}

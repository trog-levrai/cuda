#include "model.hh"

arma::Mat<float> Model::normalize_(arma::Mat<float>& matrix) {
  return matrix.each_row() - arma::mean(matrix);
}

#include "model.hh"

float Model::sigmoid_(float x) {
  return 1.0 / (1.0 + std::exp(-x));
}

float Model::dsigmoid_(float x) {
  return x * (1.0 - x);
}

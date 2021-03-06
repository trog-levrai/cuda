# pragma once

# include <cmath>
# include <iostream>
# include <functional>
# include <cuda.h>

struct activation_func {
  virtual const float (float) f() = 0;
  virtual const float (float) d_f() = 0;
};

struct tan_h : public activation_func {

  const float (float) f() {
    return [] __device__ (float a) {
      return (std::exp(2 * a) - 1) / (std::exp(2 * a) + 1);
    };
  };

  const float (float) d_f() {
    return [] __device__ (float a) {
      float h = (std::exp(2 * a) - 1) / (std::exp(2 * a) + 1);
      return 1 - h * h; 
    };
  };

};

struct relu : public activation_func {

  const float (float) f() {
    return [] __device__ (float a) {
      return a >= 0 ? a : 0;
    };
  };

  const float (float) d_f() {
    return [] __device__ (float a) {
      return a >= 0 ? 1 : 0;
    };
  };

};

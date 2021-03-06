# pragma once

# include <cmath>
# include <iostream>

struct activation_func {
  virtual const std::function<float (float)> f() = 0;
  virtual const std::function<float (float)> d_f() = 0;
};

struct tan_h : public activation_func {

  const std::function<float (float)> f() {
    return [](float a) {
      return (std::exp(2 * a) - 1) / (std::exp(2 * a) + 1);
    };
  };

  const std::function<float (float)> d_f() {
    return [](float a) {
      float h = (std::exp(2 * a) - 1) / (std::exp(2 * a) + 1);
      return 1 - h * h; 
    };
  };

};

struct relu : public activation_func {

  const std::function<float (float)> f() {
    return [](float a) {
      return a >= 0 ? a : 0;
    };
  };

  const std::function<float (float)> d_f() {
    return [](float a) {
      return a >= 0 ? 1 : 0;
    };
  };

};

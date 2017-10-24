# pragma once

# include <stdlib.h>

struct Matrix {
  Matrix(double* tab_, size_t m_, size_t n_) :
    tab(tab_), m(m_), n(n_) {}

  ~Matrix()
  {
    free(tab);
  }

  double* tab;
  size_t m;
  size_t n;
};

#include "iostream"
#include "../lib/lapacke.h"

void printMat(double* mat) {
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 2; ++j)
      std::cout << mat[i * 3 + j] << ", ";
    std::cout << mat[i * 3 + 2] << std::endl;
  }
}

int main() {
  char    TRANS = 'N';
  int     INFO=3;
  int     LDA = 3;
  int     LDB = 3;
  int     N = 3;
  int     NRHS = 1;
  int     IPIV[3] ;

  double  A[9] =
  {
    1, 2, 3,
    1, 2, 3,
    1, 2, 3
  };

  printMat(A);
}

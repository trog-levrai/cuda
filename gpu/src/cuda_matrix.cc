#include "cuda_matrix.hh"

CudaMatrix::~CudaMatrix() {
  cudaFree(a_d_);
}

CudaMatrix::CudaMatrix(cublasHandle_t handle, size_t M, size_t N, const float* a_h) {
  cublasError_t cudaStat;
  cublasStatus_t stat;
  this->handle_ = handle;
  this->M_ = M;
  this->N_ = N;
  cudaStat = cudaMalloc ((void**)&a_d_, M*N*sizeof(*a));
  if (cudaStat != cudaSuccess)
    throw std::runtime_error("Device memory allocation failed");

  stat = cublasSetMatrix (M, N, sizeof(*a_h), a, M, a_d_, M);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    cudaFree(a_d_);
    cublasDestroy(handle);
    throw std::runtime_error("data download failed");
  }
}

CudaMatrix::CudaMatrix(cublasHandle_t handle, size_t M, size_t N) {
  cublasError_t cudaStat;
  this->handle_ = handle;
  this->M_ = M;
  this->N_ = N;
  cudaStat = cudaMalloc ((void**)&a_d_, M*N*sizeof(*a));
  if (cudaStat != cudaSuccess)
    throw std::runtime_error("Device memory allocation failed");
}

CudaMatrix::CudaMatrix(CudaMatrix& m) {
  cublasError_t cudaStat;
  this->handle_ = m.handle_;
  this->M_ = m.M_;
  this->N_ = m.N_;
  cudaStat = cudaMalloc ((void**)&a_d_, M*N*sizeof(*a));
  if (cudaStat != cudaSuccess)
    throw std::runtime_error("Device memory allocation failed");

  cudaStat = cudaMemcpy(this->a_d_, m.a_d_, M * N * sizeof(*a), cudaMemcpyDeviceToDevice);
  if (cudaStat != cudaSuccess)
    throw std::runtime_error("Device Memcpy failed");
}

CudaMatrix& CudaMatrix::operator*(const CudaMatrix& m) {
  CudaMatrix c = CudaMatrix(handle_, M_, m.N_);
  float alpha = 1.;
  float beta = 0.;
  cublasStatus_t stat = cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, M_, m.N_, N_, &alpha, a_d_, M_, m.a_d_, m.N_, &beta, c.a_d_, M_);
  return c;
}

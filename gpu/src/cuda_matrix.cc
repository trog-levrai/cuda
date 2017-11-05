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
  cudaStat = cudaMalloc ((void**)&a_d_, M * N * sizeof (*a));
  if (cudaStat != cudaSuccess)
    throw std::runtime_error("Device memory allocation failed");

  stat = cublasSetMatrix (M, N, sizeof (*a_h), a, M, a_d_, M);
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
  cudaStat = cudaMalloc ((void**)&a_d_, M * N * sizeof (*a));
  if (cudaStat != cudaSuccess)
    throw std::runtime_error("Device memory allocation failed");
}

CudaMatrix::CudaMatrix(CudaMatrix& m) {
  cublasError_t cudaStat;
  this->handle_ = m.handle_;
  this->M_ = m.M_;
  this->N_ = m.N_;
  cudaStat = cudaMalloc ((void**)&a_d_, M * N * sizeof (*a));
  if (cudaStat != cudaSuccess)
    throw std::runtime_error("Device memory allocation failed");

  cudaStat = cudaMemcpy(this->a_d_, m.a_d_, M * N * sizeof (*a), cudaMemcpyDeviceToDevice);
  if (cudaStat != cudaSuccess)
    throw std::runtime_error("Device Memcpy failed");
}

CudaMatrix& CudaMatrix::operator*(const CudaMatrix& m) {
  CudaMatrix c = CudaMatrix(handle_, M_, m.N_);
  float alpha = 1.;
  float beta = 0.;
  cublasStatus_t stat = cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, M_, m.N_, N_, &alpha, a_d_, M_, m.a_d_, m.N_, &beta, c.a_d_, M_);
  if (stat != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("Matrix dot product failed");
  return c;
}

CudaMatrix& CudaMatrix::operator*(float x) {
  CudaMatrix c = CudaMatrix(handle_, M_, N_);
  cublasStatus_t stat = cublasSscal(handle_, c.M_ * c.N_, &x, c.a_d_, 1);
  if (stat != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("Matrix multiplication with scalar failed");
  return c;
}

CudaMatrix& CudaMatrix::operator%(const CudaMatrix& m) {
  CudaMatrix c = CudaMatrix(handle_, M_, m.N_);
  dim3 DimGrid((M_ * N_ - 1)/256 + 1, 1, 1);
  dim3 DimBlock(256, 1, 1);
  cudaError_t stat = vecMulKernel<<<DimGrid,DimBlock>>>(d_n_, m.d_n_, c.m_n_, M_ * N_);
  if (stat != cudaSuccess)
    throw std::runtime_error("Element-wise multiplication failed");
  return c;
}

CudaMatrix& CudaMatrix::operator+(const CudaMatrix& m) {
  CudaMatrix c = CudaMatrix(handle_, M_, m.N_);
  dim3 DimGrid((M_ * N_ - 1)/256 + 1, 1, 1);
  dim3 DimBlock(256, 1, 1);
  cudaError_t stat = vecAddKernel<<<DimGrid,DimBlock>>>(d_n_, m.d_n_, c.m_n_, M_ * N_);
  if (stat != cudaSuccess)
    throw std::runtime_error("Element-wise addition failed");
  stat = cudaDeviceSynchronize();    
  if (stat != cudaSuccess)
    throw std::runtime_error("Device synchrnization failed");
  return c;
}

CudaMatrix& CudaMatrix::operator-(const CudaMatrix& m) {
  CudaMatrix c = CudaMatrix(handle_, M_, m.N_);
  dim3 DimGrid((M_ * N_ - 1)/256 + 1, 1, 1);
  dim3 DimBlock(256, 1, 1);
  cudaError_t stat = vecSubKernel<<<DimGrid,DimBlock>>>(d_n_, m.d_n_, c.m_n_, M_ * N_);
  if (stat != cudaSuccess)
    throw std::runtime_error("Element-wise substraction failed");
  stat = cudaDeviceSynchronize();    
  if (stat != cudaSuccess)
    throw std::runtime_error("Device synchrnization failed");
  return c;
}

CudaMatrix& CudaMatrix::operator+(float m) {
  CudaMatrix c = CudaMatrix(handle_, M_, m.N_);
  dim3 DimGrid((M_ * N_ - 1)/256 + 1, 1, 1);
  dim3 DimBlock(256, 1, 1);
  cudaError_t stat = scalarAddKernel<<<DimGrid,DimBlock>>>(d_n_, m, c.m_n_, M_ * N_);
  if (stat != cudaSuccess)
    throw std::runtime_error("Addition with scalar failed");
  stat = cudaDeviceSynchronize();    
  if (stat != cudaSuccess)
    throw std::runtime_error("Device synchrnization failed");
  return c;
}

CudaMatrix& CudaMatrix::operator+(float m) {
  CudaMatrix c = CudaMatrix(handle_, M_, m.N_);
  dim3 DimGrid((M_ * N_ - 1)/256 + 1, 1, 1);
  dim3 DimBlock(256, 1, 1);
  cudaError_t stat = scalarAddKernel<<<DimGrid,DimBlock>>>(d_n_, -m, c.m_n_, M_ * N_);
  if (stat != cudaSuccess)
    throw std::runtime_error("Addition with scalar failed");
  stat = cudaDeviceSynchronize();    
  if (stat != cudaSuccess)
    throw std::runtime_error("Device synchrnization failed");
  return c;
}

CudaMatrix& CudaMatrix::t() const {
  CudaMatrix c = CudaMatrix(handle_, M_, m.N_);
  cudaStatus_t stat = cublasSgeam(handle_, CUBLAS_OP_T, CUBLAS_OP_T, M_, N_, 1., a_d_, N_, nullptr, N_, 0., c.a_d_, N_);
  if (stat != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("Matrix transposition failed");
  return c;
}

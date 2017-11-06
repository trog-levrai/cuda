# include "cuda_matrix.cuh"
# include <iostream>
# include <stdlib.h>

CudaMatrix ones(size_t M, size_t N, cublasHandle_t handle) {
  float *mat;
  cudaMalloc((void**)&mat, M * N * sizeof(float));

  CudaMatrix out(handle, M, N);
  out.setMat(mat);

  out = out * 0 + 1;

  return out;
}

CudaMatrix::~CudaMatrix() {
  cudaFree(a_d_);
}

CudaMatrix::CudaMatrix(cublasHandle_t handle, size_t M, size_t N, const float* a_h) {
  cudaError_t cudaStat;
  cublasStatus_t stat;
  this->handle_ = handle;
  this->M_ = M;
  this->N_ = N;
  cudaStat = cudaMalloc ((void**)&a_d_, M * N * sizeof (float));
  if (cudaStat != cudaSuccess)
    throw std::runtime_error("Device memory allocation failed");

  stat = cublasSetMatrix (M, N, sizeof (float), a_h, M, a_d_, M);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    cudaFree(a_d_);
    cublasDestroy(handle);
    throw std::runtime_error("data download failed");
  }
}

CudaMatrix::CudaMatrix(cublasHandle_t handle, size_t M, size_t N) {
  cudaError_t cudaStat;
  this->handle_ = handle;
  this->M_ = M;
  this->N_ = N;
  cudaStat = cudaMalloc((void**)&a_d_, M * N * sizeof (float));
  if (cudaStat != cudaSuccess)
    throw std::runtime_error("Device memory allocation failed");
}

CudaMatrix::CudaMatrix(const CudaMatrix& m) {
  cudaError_t cudaStat;
  this->handle_ = m.handle_;
  this->M_ = m.M_;
  this->N_ = m.N_;
  cudaStat = cudaMalloc((void**)&a_d_, m.M_ * m.N_ * sizeof (float));
  if (cudaStat != cudaSuccess)
    throw std::runtime_error("Device memory allocation failed");

  cudaStat = cudaMemcpy(this->a_d_, m.a_d_, m.M_ * m.N_ * sizeof (float), cudaMemcpyDeviceToDevice);
  if (cudaStat != cudaSuccess)
    throw std::runtime_error("Device Memcpy failed");
}

CudaMatrix CudaMatrix::operator*(const CudaMatrix& m) const {
  CudaMatrix c = CudaMatrix(handle_, M_, m.N_);
  float alpha = 1.;
  float beta = 0.;
  cublasStatus_t stat = cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, M_, m.N_, N_, &alpha, a_d_, M_, m.a_d_, m.N_, &beta, c.a_d_, M_);
  if (stat != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("Matrix dot product failed");
  return c;
}

CudaMatrix CudaMatrix::operator*(float x) const {
  CudaMatrix c = CudaMatrix(handle_, M_, N_);
  cublasStatus_t stat = cublasSscal(handle_, c.M_ * c.N_, &x, c.a_d_, 1);
  if (stat != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("Matrix multiplication with scalar failed");
  return c;
}

CudaMatrix CudaMatrix::operator%(const CudaMatrix& m) const {
  CudaMatrix c = CudaMatrix(handle_, M_, m.N_);
  dim3 DimGrid((M_ * N_ - 1)/256 + 1, 1, 1);
  dim3 DimBlock(256, 1, 1);
  vecMulKernel<<<DimGrid,DimBlock>>>(a_d_, m.a_d_, c.a_d_, M_ * N_);
  return c;
}

CudaMatrix CudaMatrix::operator+(const CudaMatrix& m) const {
  CudaMatrix c = CudaMatrix(handle_, M_, m.N_);
  dim3 DimGrid((M_ * N_ - 1)/256 + 1, 1, 1);
  dim3 DimBlock(256, 1, 1);
  vecAddKernel<<<DimGrid,DimBlock>>>(a_d_, m.a_d_, c.a_d_, M_ * N_);
  cudaError_t stat = cudaDeviceSynchronize();
  if (stat != cudaSuccess)
    throw std::runtime_error("Device synchrnization failed");
  return c;
}

CudaMatrix CudaMatrix::operator-(const CudaMatrix& m) const {
  CudaMatrix c = CudaMatrix(handle_, M_, m.N_);
  dim3 DimGrid((M_ * N_ - 1)/256 + 1, 1, 1);
  dim3 DimBlock(256, 1, 1);
  vecSubKernel<<<DimGrid,DimBlock>>>(a_d_, m.a_d_, c.a_d_, M_ * N_);
  cudaError_t stat = cudaDeviceSynchronize();
  if (stat != cudaSuccess)
    throw std::runtime_error("Device synchrnization failed");
  return c;
}

CudaMatrix CudaMatrix::operator+(float m) const {
  CudaMatrix c = CudaMatrix(handle_, M_, N_);
  dim3 DimGrid((M_ * N_ - 1)/256 + 1, 1, 1);
  dim3 DimBlock(256, 1, 1);
  scalarAddKernel<<<DimGrid,DimBlock>>>(a_d_, m, c.a_d_, M_ * N_);
  cudaError_t stat = cudaDeviceSynchronize();
  if (stat != cudaSuccess)
    throw std::runtime_error("Device synchrnization failed");
  return c;
}

CudaMatrix CudaMatrix::operator-(float m) const {
  CudaMatrix c = CudaMatrix(handle_, M_, N_);
  dim3 DimGrid((M_ * N_ - 1)/256 + 1, 1, 1);
  dim3 DimBlock(256, 1, 1);
  scalarAddKernel<<<DimGrid,DimBlock>>>(a_d_, -m, c.a_d_, M_ * N_);
  cudaError_t stat = cudaDeviceSynchronize();
  if (stat != cudaSuccess)
    throw std::runtime_error("Device synchrnization failed");
  return c;
}

CudaMatrix CudaMatrix::t() const {
  CudaMatrix c = CudaMatrix(handle_, N_, M_);
  float alpha = 1.;
  float beta = 0.;
  cublasStatus_t stat = cublasSgeam(handle_, CUBLAS_OP_T, CUBLAS_OP_T, M_, N_, &alpha, a_d_, N_, nullptr, &beta, M_, c.a_d_, N_);
  if (stat != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("Matrix transposition failed");
  return c;
}

CudaMatrix CudaMatrix::transform(float (*f)(float)) {
  dim3 DimGrid((this->M_ * this->N_ - 1) / 256 + 1, 1, 1);
  dim3 DimBlock(256, 1, 1);
  matTransformKernel<<<DimGrid,DimBlock>>>(a_d_, f, this->M_ * this->N_);
  cudaError_t stat = cudaDeviceSynchronize();
  if (stat != cudaSuccess)
    throw std::runtime_error("Device synchrnization failed");
  return *this;
}

CudaMatrix CudaMatrix::relu() {
  dim3 DimGrid((this->M_ * this->N_ - 1) / 256 + 1, 1, 1);
  dim3 DimBlock(256, 1, 1);
  matRelu<<<DimGrid,DimBlock>>>(a_d_, this->M_ * this->N_);
  cudaError_t stat = cudaDeviceSynchronize();
  if (stat != cudaSuccess)
    throw std::runtime_error("Device synchrnization failed");
  return *this;
}

CudaMatrix CudaMatrix::d_relu() {
  dim3 DimGrid((this->M_ * this->N_ - 1) / 256 + 1, 1, 1);
  dim3 DimBlock(256, 1, 1);
  matDRelu<<<DimGrid,DimBlock>>>(a_d_, this->M_ * this->N_);
  cudaError_t stat = cudaDeviceSynchronize();
  if (stat != cudaSuccess)
    throw std::runtime_error("Device synchrnization failed");
  return *this;
}

CudaMatrix CudaMatrix::reshape(size_t M, size_t N) {
  if (M_ * N_ != M * N)
    throw std::runtime_error("Bad Reshape");
  CudaMatrix out(*this);
  out.M_ = M;
  out.N_ = N;
  return out;
}

void CudaMatrix::randomize() {
  curandState_t* states;
  cudaError_t cudaStat = cudaMalloc((void**) &states, M_ * N_ * sizeof (curandState_t));
  if (cudaStat != cudaSuccess)
    throw std::runtime_error("Device memory allocation failed");

  init<<<M_ * N_, 1>>>(time(0), states);
  dim3 DimGrid((M_ * N_ - 1)/256 + 1, 1, 1);
  dim3 DimBlock(256, 1, 1);
  randomizeKernel<<<DimGrid,DimBlock>>>(states, a_d_, M_ * N_);
}

CudaMatrix CudaMatrix::rows(size_t start, size_t end) const {
  CudaMatrix c = CudaMatrix(handle_, start - end, N_);
  dim3 DimGrid((M_ * N_ - 1)/256 + 1, 1, 1);
  dim3 DimBlock(256, 1, 1);
  rowGetter<<<DimGrid,DimBlock>>>(a_d_, c.a_d_, start, end, N_);
  cudaError_t stat = cudaDeviceSynchronize();
  if (stat != cudaSuccess)
    throw std::runtime_error("Device synchrnization failed");
  return c;
}

CudaMatrix CudaMatrix::rows(std::vector<size_t>& indices) const {
  CudaMatrix c = CudaMatrix(handle_, indices.size(), N_);
  dim3 DimGrid((M_ * N_ - 1)/256 + 1, 1, 1);
  dim3 DimBlock(256, 1, 1);
  for (size_t i = 0; i < indices.size(); ++i)
    rowGetter<<<DimGrid,DimBlock>>>(a_d_, c.a_d_ + i * N_, indices[i], indices[i] + 1, N_);
  return c;
}

float CudaMatrix::accu() const {
  return thrust::reduce(a_d_, a_d_ + M_ * N_);
}

void CudaMatrix::addBias() {
  float* newi;
  cudaError_t cudaStat = cudaMalloc((void**) &newi, this->M_ * (this->N_ + 1) * sizeof(float));
  if (cudaStat != cudaSuccess)
    throw std::runtime_error("Device memory allocation failed");

  auto tmp = a_d_;
  this->a_d_ = newi;
  this->N_++;

  *this = *this * 0 + 1;

  cudaStat = cudaMemcpy((void*)newi, (void*)tmp, this->M_ * (this->N_ - 1) * sizeof(float), cudaMemcpyDeviceToDevice);
  if (cudaStat != cudaSuccess)
    throw std::runtime_error("Memcpy failed");

  cudaFree(tmp);
}

void CudaMatrix::print() {
  float* tmp = (float*)malloc(M_ * N_ * sizeof(float));
  cublasGetMatrix(M_, N_, sizeof(float), a_d_, M_, (void *)tmp, M_);
  for (size_t i = 0; i < M_; ++i) {
    for (size_t j = 0; j < N_; ++j) {
      std::cout << tmp[i * N_ +j] << " ";
    }
    std::cout << "\n";
  }
}

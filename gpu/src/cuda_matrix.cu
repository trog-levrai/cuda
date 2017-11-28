# include "cuda_matrix.cuh"
# include "cuda_error.hh"

# include <iostream>
# include <stdlib.h>
# include <cmath>

void sync_device() {
  cudaError_t stat_ = cudaDeviceSynchronize();
  if (stat_ != cudaSuccess)
    throw std::runtime_error("Device synchronization failed");
}

CudaMatrix ones(size_t M, size_t N, cublasHandle_t handle) {
  auto out = CudaMatrix(handle, M, N);
  cudaMemset((void**)out.getMat().get(), 0, M * N * sizeof (float));
  out += 1.;

  return out;
}

CudaMatrix::~CudaMatrix() { }

void CudaMatrix::alloc() {
  if (half_) {
    half *a_d_tmp;
    CudaSafeCall(cudaMalloc((void**)&a_d_tmp, M_ * N_ * sizeof (half)));
    a_d_ = std::shared_ptr<half>(a_d_tmp, cudaFree);
  } else {
    float *a_d_tmp;
    CudaSafeCall(cudaMalloc((void**)&a_d_tmp, M_ * N_ * sizeof (float)));
    f_d_ = std::shared_ptr<float>(a_d_tmp, cudaFree);
  }
}

CudaMatrix::CudaMatrix(cublasHandle_t handle, size_t M, size_t N, const float* a_h, bool half = false) {
  this->handle_ = handle;
  this->M_ = M;
  this->N_ = N;
  this->half_ = half;
  this->alloc();

  if (half_) {
    float *a_d_tmp;
    CudaSafeCall(cudaMalloc((void**)&a_d_tmp, M_ * N_ * sizeof (float)));
    CublasSafeCall(cublasSetMatrix(M, N, sizeof (float), a_h, M, a_d_tmp, M));

    dim3 DimGrid(std::ceil((M_ * N_) / 256.0), 1, 1);
    dim3 DimBlock(256, 1, 1);
    f2h<<<DimGrid,DimBlock>>>(a_d_tmp, a_d_.get(), M_ * N_);
    CudaSafeCall(cudaFree(a_d_tmp));
    sync_device();
  }
  else
    CublasSafeCall(cublasSetMatrix(M, N, sizeof (float), a_h, M, a_d_.get(), M));
}

CudaMatrix::CudaMatrix(cublasHandle_t handle, size_t M, size_t N, bool half = false) {
  this->handle_ = handle;
  this->M_ = M;
  this->N_ = N;
  this->half_ = half;
  this->alloc();
}

CudaMatrix::CudaMatrix(const CudaMatrix& m) {
  this->handle_ = m.handle_;
  this->M_ = m.M_;
  this->N_ = m.N_;
  this->half_ = m.half_;
  this->alloc();

  if (half_)
    CudaSafeCall(cudaMemcpy(this->a_d_.get(), m.a_d_.get(), m.M_ * m.N_ * sizeof (half), cudaMemcpyDeviceToDevice));
  else
    CudaSafeCall(cudaMemcpy(this->f_d_.get(), m.f_d_.get(), m.M_ * m.N_ * sizeof (float), cudaMemcpyDeviceToDevice));
}

// WORK
CudaMatrix CudaMatrix::operator*(const CudaMatrix& m) const {
  float alpha = 1.;
  float beta = 0.;
  auto c = CudaMatrix(handle_, M_, m.N_);

  CublasSafeCall(cublasHgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m.N_, M_, N_, &alpha, m.a_d_.get(), m.N_, a_d_.get(), N_, &beta, c.a_d_.get(), m.N_));

  sync_device();

  return c;
}

// WORK
CudaMatrix CudaMatrix::mult_buff(const CudaMatrix& m, CudaMatrix& o) const {
  float alpha = 1.;
  float beta = 0.;

  o.M_ = this->M_;
  o.N_ = m.N_;

  CublasSafeCall(cublasHgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m.N_, M_, N_, &alpha, m.a_d_.get(), m.N_, a_d_.get(), N_, &beta, o.a_d_.get(), m.N_));

  sync_device();

  return o;
}

// WORK
CudaMatrix CudaMatrix::dot(const CudaMatrix& m, float alpha) const {
  float beta = 0.;
  auto c = CudaMatrix(handle_, M_, m.N_);

  CublasSafeCall(cublasHgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m.N_, M_, N_, &alpha, m.a_d_.get(), m.N_, a_d_.get(), N_, &beta, c.a_d_.get(), m.N_));

  sync_device();

  return c;
}

// WORK
CudaMatrix CudaMatrix::operator=(const CudaMatrix& m) {

  this->M_ = m.M_;
  this->N_ = m.N_;
  this->a_d_ = m.a_d_;
  this->f_d_ = m.f_d_;
  this->half_ = m.half_;

  return *this;
}

// WORK
CudaMatrix CudaMatrix::operator*(float x) const {
  auto c = CudaMatrix(*this);
  CublasSafeCall(cublasSscal(handle_, c.M_ * c.N_, &x, c.a_d_.get(), 1));

  sync_device();

  return c;
}

// WORK
CudaMatrix CudaMatrix::operator%(const CudaMatrix& m) const {
  if (this->shape() != m.shape()) {
    std::cout << "% failed\n";
    this->print_shape("this\t");
    m.print_shape("m\t");
    exit(-1);
  }

  auto c = CudaMatrix(handle_, M_, m.N_);
  dim3 DimGrid(std::ceil((M_ * N_) / 256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);
  vecMulKernel<<<DimGrid,DimBlock>>>(a_d_.get(), m.a_d_.get(), c.a_d_.get(), M_ * N_);

  sync_device();

  return c;
}

// WORK
CudaMatrix CudaMatrix::operator%=(const CudaMatrix& m) {
  if (this->shape() != m.shape()) {
    std::cout << "% failed\n";
    this->print_shape("this\t");
    m.print_shape("m\t");
    exit(-1);
  }

  dim3 DimGrid(std::ceil((M_ * N_) / 256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);
  vecMulKernel<<<DimGrid,DimBlock>>>(a_d_.get(), m.a_d_.get(), a_d_.get(), M_ * N_);

  sync_device();

  return *this;
}

// WORK
CudaMatrix CudaMatrix::operator+(const CudaMatrix& m) const {
  if (this->shape() != m.shape()) {
    std::cout << "+ failed\n";
    this->print_shape("this\t");
    m.print_shape("m\t");
  }

  CudaMatrix c = CudaMatrix(handle_, m.M_, m.N_);

  dim3 DimGrid(std::ceil((M_ * N_) / 256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);
  vecAddKernel<<<DimGrid,DimBlock>>>(a_d_.get(), m.a_d_.get(), c.a_d_.get(), M_ * N_);

  sync_device();

  return c;
}

// WORK
CudaMatrix CudaMatrix::operator+=(const CudaMatrix& m) {
  if (this->shape() != m.shape()) {
    std::cout << "+ failed\n";
    this->print_shape("this\t");
    m.print_shape("m\t");
  }

  dim3 DimGrid(std::ceil((M_ * N_) / 256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);
  vecAddKernel<<<DimGrid,DimBlock>>>(a_d_.get(), m.a_d_.get(), a_d_.get(), M_ * N_);

  sync_device();

  return *this;
}

// WORK
CudaMatrix CudaMatrix::operator-(const CudaMatrix& m) const {
  dim3 DimGrid(std::ceil((N_ * M_) / 256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);

  auto c = CudaMatrix(handle_, M_, N_);
  vecSubKernel<<<DimGrid,DimBlock>>>(a_d_.get(), m.a_d_.get(), c.a_d_.get(), M_ * N_);
  sync_device();
  return c;
}

// WORK
CudaMatrix CudaMatrix::operator-=(const CudaMatrix& m) {
  dim3 DimGrid(std::ceil((N_ * M_) / 256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);

  vecSubKernel<<<DimGrid,DimBlock>>>(a_d_.get(), m.a_d_.get(), a_d_.get(), M_ * N_);
  sync_device();
  return *this;
}

// WORK
CudaMatrix CudaMatrix::operator+(float m) const {
  auto c = CudaMatrix(handle_, M_, N_);

  dim3 DimGrid(std::ceil((M_ * N_) / 256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);
  scalarAddKernel<<<DimGrid,DimBlock>>>(a_d_.get(), m, c.a_d_.get(), M_ * N_);

  sync_device();

  return c;
}

CudaMatrix CudaMatrix::operator+=(float m) {
  dim3 DimGrid(std::ceil((M_ * N_) / 256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);
  scalarAddKernel<<<DimGrid,DimBlock>>>(a_d_.get(), m, a_d_.get(), M_ * N_);

  sync_device();

  return *this;
}

// WORK
CudaMatrix CudaMatrix::operator-(float m) const {
  auto c = CudaMatrix(handle_, M_, N_);

  dim3 DimGrid(std::ceil((M_ * N_) / 256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);
  scalarAddKernel<<<DimGrid,DimBlock>>>(a_d_.get(), -m, c.a_d_.get(), M_ * N_);

  sync_device();

  return c;
}

// WORK
CudaMatrix CudaMatrix::t() const {
  auto c = CudaMatrix(handle_, N_, M_);

  float alpha = 1.;
  float beta = 0.;

  CublasSafeCall(cublasHgeam(handle_, CUBLAS_OP_T, CUBLAS_OP_T, M_, N_, &alpha, this->a_d_.get(), N_, &beta, this->a_d_.get(), N_, c.a_d_.get(), M_));

  sync_device();
  return c;
}

CudaMatrix CudaMatrix::transform(float (*f)(float)) {
  dim3 DimGrid(std::ceil((M_ * N_) / 256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);
  matTransformKernel<<<DimGrid,DimBlock>>>(a_d_.get(), f, this->M_ * this->N_);

  sync_device();

  return *this;
}

// WORK
CudaMatrix CudaMatrix::relu() {
  dim3 DimGrid(std::ceil((M_ * N_) / 256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);
  matTanh<<<DimGrid,DimBlock>>>(a_d_.get(), this->M_ * this->N_);

  sync_device();

  return *this;
}

// WORK
CudaMatrix CudaMatrix::d_relu() {
  dim3 DimGrid(std::ceil((M_ * N_) / 256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);
  matDTanh<<<DimGrid,DimBlock>>>(a_d_.get(), this->M_ * this->N_);

  sync_device();

  return *this;
}

// WORK
CudaMatrix CudaMatrix::reshape(size_t M, size_t N) {
  if (M_ * N_ != M * N)
    throw std::runtime_error("Bad Reshape");

  CudaMatrix out = CudaMatrix(*this);
  out.M_ = M;
  out.N_ = N;

  return out;
}

// WORK
void CudaMatrix::randomize() {
  curandState_t* states;
  CudaSafeCall(cudaMalloc((void**) &states, M_ * N_ * sizeof (curandState_t)));

  init<<<M_ * N_, 1>>>(time(0), states);

  sync_device();

  dim3 DimGrid(std::ceil((M_ * N_) / 256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);
  randomizeKernel<<<DimGrid,DimBlock>>>(states, a_d_.get(), M_ * N_);

  sync_device();
}

// WORK
CudaMatrix CudaMatrix::rows(size_t start, size_t end) const {
  auto c = CudaMatrix(handle_, end - start, N_);

  dim3 DimGrid(std::ceil((M_ * N_) / 256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);
  rowGetter<<<DimGrid,DimBlock>>>(a_d_.get(), c.a_d_.get(), start, end, N_);

  sync_device();

  return c;
}

// WORK
CudaMatrix CudaMatrix::rows(std::vector<size_t>& indices) const {
  auto c = CudaMatrix(handle_, indices.size(), N_);

  dim3 DimGrid(std::ceil((M_ * N_) / 256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);
  for (size_t i = 0; i < indices.size(); ++i)
    rowGetter<<<DimGrid,DimBlock>>>(a_d_.get(), c.a_d_.get() + i * N_, indices[i], indices[i] + 1, N_);

  sync_device();

  return c;
}

// WORK
float CudaMatrix::accu() const {
  float *a_d_tmp;
  CudaSafeCall(cudaMalloc((void**)&a_d_tmp, M_ * N_ * sizeof (float)));
  dim3 DimGrid(std::ceil((M_ * N_) / 256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);
  h2f<<<DimGrid,DimBlock>>>(a_d_get, a_d_tmp, M_ * N_);
  CudaSafeCall(cudaFree(a_d_tmp));
  sync_device();
  return thrust::reduce(thrust::device, a_d_.get(), a_d_.get() + M_ * N_);
}

// WORK
CudaMatrix CudaMatrix::addBias() {
  auto out = ones(this->M_, this->N_ + 1, handle_);

  for (size_t i = 0; i < this->M_; ++i)
    CudaSafeCall(cudaMemcpy(out.a_d_.get() + i * (this->N_ + 1), this->a_d_.get() + i * N_, N_ * sizeof (float), cudaMemcpyDeviceToDevice));

  sync_device();

  return out;
}

void CudaMatrix::print() const {
  float* tmp = (float*)malloc(M_ * N_ * sizeof (float));
  cublasGetMatrix(M_, N_, sizeof (float), a_d_.get(), M_, (void *)tmp, M_);
  for (size_t i = 0; i < M_; ++i) {
    for (size_t j = 0; j < N_; ++j) {
      std::cout << tmp[i * N_ +j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
  free(tmp);
}

std::pair<size_t, size_t> CudaMatrix::shape() const {
  return std::pair<size_t, size_t>(M_, N_);
}

void CudaMatrix::print_shape(std::string str) const {
  std::cout << str << this->M_ << ":" << this->N_ << std::endl;
}

CudaMatrix CudaMatrix::getHalf() const {
  if (half_) {
    std::cout << "This is a half precision matrix." << std::endl;
    exit(1);
  }
  auto ans = CudaMatrix(handle_, M_, N_, true);
  dim3 DimGrid(std::ceil((M_ * N_) / 256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);
  f2h<<<DimGrid,DimBlock>>>(f_d_.get(), aux.a_d_.get(), M_ * N_);
  CudaSafeCall(cudaFree(a_d_tmp));
  sync_device();
}

CudaMatrix CudaMatrix::getHalf() const {
  if (!half_) {
    std::cout << "This is a sigle precision matrix." << std::endl;
    exit(1);
  }
  auto ans = CudaMatrix(handle_, M_, N_, false);
  dim3 DimGrid(std::ceil((M_ * N_) / 256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);
  h2f<<<DimGrid,DimBlock>>>(a_d_.get(), aux.f_d_.get(), M_ * N_);
  CudaSafeCall(cudaFree(a_d_tmp));
  sync_device();
}

# include "cuda_matrix.cuh"
# include "cuda_error.hh"

# include <iostream>
# include <stdlib.h>
# include <cmath>

void sync_device() {
  cudaError_t stat_ = cudaDeviceSynchronize();
  if (stat_ != cudaSuccess)
    throw std::runtime_error("Device synchrnization failed");
}

CudaMatrix& ones(size_t M, size_t N, cublasHandle_t handle) {
  CudaMatrix* out = new CudaMatrix(handle, M, N);
  cudaMemset((void**)out->getMat().get(), 0, M * N * sizeof(float));
  *out = *out + 1.;

  return *out;
}

CudaMatrix::~CudaMatrix() {
}

CudaMatrix::CudaMatrix(cublasHandle_t handle, size_t M, size_t N, const float* a_h) {
  this->handle_ = handle;
  this->M_ = M;
  this->N_ = N;
  float *a_d_tmp;

  CudaSafeCall(cudaMalloc ((void**)&a_d_tmp, M * N * sizeof (float)));

  this->a_d_ = std::shared_ptr<float>(a_d_tmp, cudaFree);

  CublasSafeCall(cublasSetMatrix(M, N, sizeof (float), a_h, M, a_d_.get(), M));
}

CudaMatrix::CudaMatrix(cublasHandle_t handle, size_t M, size_t N) {
  this->handle_ = handle;
  this->M_ = M;
  this->N_ = N;
  float *a_d_tmp;
  CudaSafeCall(cudaMalloc((void**)&a_d_tmp, M * N * sizeof (float)));
  a_d_ = std::shared_ptr<float>(a_d_tmp, cudaFree);
}

CudaMatrix::CudaMatrix(const CudaMatrix& m) {
  this->handle_ = m.handle_;
  this->M_ = m.M_;
  this->N_ = m.N_;
  float *a_d_tmp;

  CudaSafeCall(cudaMalloc((void**)&a_d_tmp, m.M_ * m.N_ * sizeof (float)));
  this->a_d_ = std::shared_ptr<float>(a_d_tmp, cudaFree);

  CudaSafeCall(cudaMemcpy(this->a_d_.get(), m.a_d_.get(), m.M_ * m.N_ * sizeof (float), cudaMemcpyDeviceToDevice));
}

// WORK
CudaMatrix& CudaMatrix::dot(const CudaMatrix& m, float al) const {
  float *a;
  float *b;
  float *c;
  
  CudaSafeCall(cudaMalloc((void**)&a, M_ * N_ * sizeof (float)));
  CudaSafeCall(cudaMalloc((void**)&b, m.M_ * m.N_ * sizeof (float)));
  CudaSafeCall(cudaMalloc((void**)&c, M_ * m.N_ * sizeof (float)));

  dim3 DimGrid(std::ceil((M_ * N_) / 256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);
  rowToCol<<<DimGrid,DimBlock>>>(a_d_.get(), a, N_, M_);

  DimGrid = dim3(std::ceil((m.M_ * m.N_) / 256.0), 1, 1);
  DimBlock = dim3(256, 1, 1);
  rowToCol<<<DimGrid,DimBlock>>>(m.a_d_.get(), b, m.N_, m.M_);

  int m_ = M_;
  int k_ = N_;
  int n_ = m.N_;

  int lda = m_;
  int ldb = k_;
  int ldc = m_;
  CudaMatrix* out = new CudaMatrix(handle_, m_, n_);

  float be = 0.;

  float* alpha = &al;
  float* beta = &be;

  sync_device();

  CublasSafeCall(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m_, n_, k_, alpha, a, lda, b, ldb, beta, c, ldc));

  sync_device();

  DimGrid = dim3(std::ceil((M_ * m.N_) / 256.0), 1, 1);
  DimBlock = dim3(256, 1, 1);
  colToRow<<<DimGrid,DimBlock>>>(c, out->a_d_.get(), m.N_, M_);
  
  sync_device();

  return *out;
}

// WORK
CudaMatrix& CudaMatrix::operator*(const CudaMatrix& m) const {
  return this->dot(m, 1.0);
}

// WORK
CudaMatrix& CudaMatrix::operator=(const CudaMatrix& m) {
  this->M_ = m.M_;
  this->N_ = m.N_;
  this->a_d_ = std::shared_ptr<float>(m.getMat());

  CudaSafeCall(cudaMemcpy(this->a_d_.get(), m.a_d_.get(), m.M_ * m.N_ * sizeof (float), cudaMemcpyDeviceToDevice));
  
  return *this;
}

// WORK
CudaMatrix& CudaMatrix::operator*(float x) const {
  CudaMatrix *c = new CudaMatrix(*this);
  CublasSafeCall(cublasSscal(handle_, c->M_ * c->N_, &x, c->a_d_.get(), 1));

  sync_device();

  return *c;
}

// WORK
CudaMatrix& CudaMatrix::operator%(const CudaMatrix& m) const {
  if (this->shape() != m.shape()) {
    this->print_shape("this\t");
    m.print_shape("m\t");
    exit(-1);
  }

  CudaMatrix *c = new CudaMatrix(handle_, M_, m.N_);
  dim3 DimGrid(std::ceil((M_ * N_) / 256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);
  vecMulKernel<<<DimGrid,DimBlock>>>(a_d_.get(), m.a_d_.get(), c->a_d_.get(), M_ * N_);

  sync_device();

  return *c;
}

// WORK
CudaMatrix& CudaMatrix::operator+(const CudaMatrix& m) const {
  if (this->shape() != m.shape()) {
    this->print_shape("this\t");
    m.print_shape("m\t");
  }

  CudaMatrix *c = new CudaMatrix(handle_, m.M_, m.N_);

  dim3 DimGrid(std::ceil((M_ * N_) / 256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);
  vecAddKernel<<<DimGrid,DimBlock>>>(a_d_.get(), m.a_d_.get(), c->a_d_.get(), M_ * N_);

  sync_device();

  return *c;
}

// WORK
CudaMatrix& CudaMatrix::operator-(const CudaMatrix& m) const {
  CudaMatrix* c;

  // FIXME
  //This case is vector to matrix.
  if (this->shape() != m.shape()) {
    this->print_shape("this\t");
    m.print_shape("m\t");
    if (M_ == 1) {
      //This instance is a rowvec
      c = new CudaMatrix(handle_, m.M_, m.N_);

      // FIXME

      dim3 DimGrid(std::ceil((N_) / 256.0), 1, 1);
      dim3 DimBlock(256, 1, 1);

      for (size_t i = 0; i < m.M_; ++i)
        vecSubKernel<<<DimGrid,DimBlock>>>(a_d_.get(), m.a_d_.get() + i * m.N_, c->a_d_.get() + i * m.N_, N_);

      sync_device();
    } else if (m.M_ == 1) {
      //m instance is a rowvec
      c = new CudaMatrix(handle_, M_, N_);

      dim3 DimGrid(std::ceil((N_) / 256.0), 1, 1);
      dim3 DimBlock(256, 1, 1);

      // FIXME
      for (size_t i = 0; i < M_; ++i) {
        vecSubKernel<<<DimGrid,DimBlock>>>(a_d_.get() + i * N_, m.a_d_.get(), c->a_d_.get() + i * N_, N_);
      }
      sync_device();
    } else {
      return (this->t() - m.t()).t();
    }
  } else {
    dim3 DimGrid(std::ceil((N_ * M_) / 256.0), 1, 1);
    dim3 DimBlock(256, 1, 1);

    c = new CudaMatrix(handle_, M_, N_);
    vecSubKernel<<<DimGrid,DimBlock>>>(a_d_.get(), m.a_d_.get(), c->a_d_.get(), M_ * N_);
    sync_device();
  }
  return *c;
}

// WORK
CudaMatrix& CudaMatrix::operator+(float m) const {
  CudaMatrix* c = new CudaMatrix(handle_, M_, N_);

  dim3 DimGrid(std::ceil((M_ * N_) / 256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);
  scalarAddKernel<<<DimGrid,DimBlock>>>(a_d_.get(), m, c->a_d_.get(), M_ * N_);

  sync_device();

  return *c;
}

// WORK
CudaMatrix& CudaMatrix::operator-(float m) const {
  CudaMatrix* c = new CudaMatrix(handle_, M_, N_);

  dim3 DimGrid(std::ceil((M_ * N_) / 256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);
  scalarAddKernel<<<DimGrid,DimBlock>>>(a_d_.get(), -m, c->a_d_.get(), M_ * N_);

  sync_device();

  return *c;
}

// WORK
CudaMatrix& CudaMatrix::t() const {
  CudaMatrix* c = new CudaMatrix(handle_, N_, M_);

  float alpha = 1.;
  float beta = 0.;
  cublasStatus_t stat = cublasSgeam(handle_, CUBLAS_OP_T, CUBLAS_OP_T, M_, N_, &alpha, this->a_d_.get(), N_, &beta, this->a_d_.get(), N_, c->a_d_.get(), M_);

  sync_device();

  if (stat != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("Matrix transposition failed");
  return *c;
}

CudaMatrix& CudaMatrix::transform(float (*f)(float)) {
  dim3 DimGrid(std::ceil((M_ * N_) / 256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);
  matTransformKernel<<<DimGrid,DimBlock>>>(a_d_.get(), f, this->M_ * this->N_);

  sync_device();

  return *this;
}

// WORK
CudaMatrix& CudaMatrix::relu() {
  dim3 DimGrid(std::ceil((M_ * N_) / 256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);
  matTanh<<<DimGrid,DimBlock>>>(a_d_.get(), this->M_ * this->N_);

  sync_device();

  return *this;
}

// WORK
CudaMatrix& CudaMatrix::d_relu() {
  dim3 DimGrid(std::ceil((M_ * N_) / 256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);
  matDTanh<<<DimGrid,DimBlock>>>(a_d_.get(), this->M_ * this->N_);

  sync_device();

  return *this;
}

// WORK
CudaMatrix& CudaMatrix::reshape(size_t M, size_t N) {
  if (M_ * N_ != M * N)
    throw std::runtime_error("Bad Reshape");

  CudaMatrix *out = new CudaMatrix(*this);
  out->M_ = M;
  out->N_ = N;

  return *out;
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
CudaMatrix& CudaMatrix::rows(size_t start, size_t end) const {
  CudaMatrix* c = new CudaMatrix(handle_, end - start, N_);

  dim3 DimGrid(std::ceil((M_ * N_) / 256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);
  rowGetter<<<DimGrid,DimBlock>>>(a_d_.get(), c->a_d_.get(), start, end, N_);

  sync_device();

  return *c;
}

// WORK
CudaMatrix& CudaMatrix::rows(std::vector<size_t>& indices) const {
  CudaMatrix* c = new CudaMatrix(handle_, indices.size(), N_);

  dim3 DimGrid(std::ceil((M_ * N_) / 256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);
  for (size_t i = 0; i < indices.size(); ++i)
    rowGetter<<<DimGrid,DimBlock>>>(a_d_.get(), c->a_d_.get() + i * N_, indices[i], indices[i] + 1, N_);

  sync_device();

  return *c;
}

// WORK
float CudaMatrix::accu() const {
  return thrust::reduce(thrust::device, a_d_.get(), a_d_.get() + M_ * N_);
}

// WORK
CudaMatrix& CudaMatrix::addBias() {
  CudaMatrix& out = ones(this->M_, this->N_ + 1, handle_);

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
}

std::pair<size_t, size_t> CudaMatrix::shape() const {
  return std::pair<size_t, size_t>(M_, N_);
}

void CudaMatrix::print_shape(std::string str) const {
  std::cout << str << this->M_ << ":" << this->N_ << std::endl;
}

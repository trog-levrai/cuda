# include "matrix_helper.cuh"
# include <stdio.h>

__global__ void vecMulKernel(float* A, float* B, float* C, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i<n) C[i] = A[i] * B[i];
}

__global__ void vecAddKernel(float* A, float* B, float* C, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i<n) C[i] = A[i] + B[i];
}

__global__ void vecSubKernel(float* A, float* B, float* C, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i<n) C[i] = A[i] - B[i];
}

__global__ void scalarAddKernel(float* A, float s, float* C, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i<n) {
    C[i] = A[i] + s;
  }
}

__global__ void matTransformKernel(float* A, float (*f)(float), int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i<n) A[i] = f(A[i]);
}

__global__ void init(unsigned int seed, curandState_t* states) {
  curand_init(seed, blockIdx.x, 0, &states[blockIdx.x]);
}

__global__ void randomizeKernel(curandState_t* states, float* a, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i<n) a[i] = curand_uniform(&states[i]);
}

__global__ void rowGetter(float* src, float* dest, size_t first, size_t last, size_t col) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < last * col && i >= first * col && i < last * col)
    dest[i - first * col] = src[i];
}

__global__ void matRelu(float* a, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i<n) a[i] = a[i] < 0 ? 0 : a[i];
}

__global__ void matTanh(float* a, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    float ex = expf(2. * a[i]);
    a[i] = (ex - 1) / (ex + 1);
  }
}

__global__ void matDRelu(float* a, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i<n) a[i] = a[i] < 0 ? 0 : 1;
}

__global__ void matDTanh(float* a, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    float ex = expf(2. * a[i]);
    float x = (ex - 1) / (ex + 1);
    a[i] = 1. - x * x;
  }
}

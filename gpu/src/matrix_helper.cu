# include "matrix_helper.cuh"
# include <stdio.h>

__global__ void vecMulKernel(half* A, half* B, half* C, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i<n) C[i] = A[i] * B[i];
}

__global__ void vecAddKernel(half* A, half* B, half* C, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i<n) C[i] = A[i] + B[i];
}

__global__ void vecSubKernel(half* A, half* B, half* C, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i<n) C[i] = A[i] - B[i];
}

__global__ void scalarAddKernel(half* A, float s, half* C, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i<n) {
    half h = __float2half(s);
    C[i] = A[i] + h;
  }
}

__global__ void matTransformKernel(half* A, half (*f)(half), int n) {
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

__global__ void rowGetter(half* src, half* dest, size_t first, size_t last, size_t col) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < last * col && i >= first * col && i < last * col)
    dest[i - first * col] = src[i];
}

__global__ void matRelu(half* a, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i<n) a[i] = a[i] < 0 ? 0 : a[i];
}

__global__ void matTanh(half* a, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    float x = __half2float(a[i]);
    x /= SF;
    x = tanhf(x) * SF;
    a[i] = __float2half(x);
  }
}

__global__ void matDRelu(half* a, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i<n) a[i] = a[i] < 0 ? 0 : 1;
}

__global__ void matDTanh(half* a, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    float x = __half2float(a[i]);
    x /= SF;
    x = tanhf(x) * SF;
    half h = __float2half2(x);
    a[i] = 1. - h * h;
  }
}
__device__ void f2h(float* src, half* dst, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i<n) dst[i] = __float2half(src[i] * SF);
}
__device__ void h2f(half* src, float* dst, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i<n) dst[i] = __half2float(src[i]) / SF;
}

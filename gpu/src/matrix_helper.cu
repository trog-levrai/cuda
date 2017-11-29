# include "matrix_helper.cuh"
# include <stdio.h>

__global__ void vecMulKernel(void* A, void* B, void* C, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  half* a = (half*)A;
  half* b = (half*)B;
  half* c = (half*)C;
  if (i<n) c[i] = a[i] * b[i];
}

__global__ void vecAddKernel(void* A, void* B, void* C, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  half* a = (half*)A;
  half* b = (half*)B;
  half* c = (half*)C;
  if (i<n) c[i] = a[i] + b[i];
}

__global__ void vecSubKernel(void* A, void* B, void* C, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  half* a = (half*)A;
  half* b = (half*)B;
  half* c = (half*)C;
  if (i<n) c[i] = a[i] - b[i];
}

__global__ void scalarAddKernel(void* A, float s, void* C, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  half* a = (half*)A;
  half* c = (half*)C;
  if (i<n) {
    half h = __float2half(s);
    c[i] = a[i] + h;
  }
}

__global__ void matTransformKernel(void* A, half (*f)(half), int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  half* a = (half*)A;
  if (i<n) a[i] = f(a[i]);
}

__global__ void init(unsigned int seed, curandState_t* states) {
  curand_init(seed, blockIdx.x, 0, &states[blockIdx.x]);
}

__global__ void randomizeKernel(curandState_t* states, float* a, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i<n) a[i] = curand_uniform(&states[i]);
}

__global__ void rowGetter(void* src, void* dest, size_t first, size_t last, size_t col) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  half* s = (half*)src;
  half* d = (half*)dest;
  if (i < last * col && i >= first * col && i < last * col)
    d[i - first * col] = s[i];
}

__global__ void matRelu(void* a, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  half* A = (half*)a;
  if (i<n) A[i] = A[i] < 0 ? 0 : a[i];
}

__global__ void matTanh(void* a, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    half* A = (half*)a;
    float x = __half2float(A[i]);
    x /= SF;
    x = tanhf(x) * SF;
    A[i] = __float2half(x);
  }
}

__global__ void matDRelu(void* a, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  half* A = (half*)a;
  if (i<n) A[i] = A[i] < 0 ? 0 : 1;
}

__global__ void matDTanh(void* a, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    half* A = (half*)a;
    float x = __half2float(a[i]);
    x /= SF;
    x = tanhf(x) * SF;
    half h = __float2half2(x);
    a[i] = 1. - h * h;
  }
}
__global__ void f2h(float* src, void* dst, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  half* d = (half*)dst;
  if (i<n) d[i] = __float2half(src[i] * SF);
}
__global__ void h2f(void* src, float* dst, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  half* s = (half*)src;
  if (i<n) dst[i] = __half2float(s[i]) / SF;
}

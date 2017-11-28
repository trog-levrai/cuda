# pragma once

# include <functional>
# include <curand.h>
# include <curand_kernel.h>
# include <cuda_fp16.h>

# define SF 128.

__global__ void vecMulKernel(half* a, half* b, half* c, int n);
__global__ void vecAddKernel(half* a, half* b, half* c, int n);
__global__ void vecSubKernel(half* a, half* b, half* c, int n);
__global__ void scalarAddKernel(half*, float, half*, int);
__global__ void matTransformKernel(half* a, half (*f)(half), int n);
__global__ void matRelu(half* a, int n);
__global__ void matDRelu(half* a, int n);
__global__ void matTanh(half* a, int n);
__global__ void matDTanh(half* a, int n);
__global__ void init(unsigned int seed, curandState_t* states);
__global__ void randomizeKernel(curandState_t* states, float* a, int n);
__global__ void rowGetter(half* src, half* dest, size_t first, size_t last, size_t col);
__device__ void f2h(float*, half*, int n);
__device__ void h2f(half*, float*, int n);

# pragma once

# include <functional>
# include <curand.h>
# include <curand_kernel.h>
# include <cuda_fp16.h>

# define SF 128.

__global__ void vecMulKernel(void* a, void* b, void* c, int n);
__global__ void vecAddKernel(void* a, void* b, void* c, int n);
__global__ void vecSubKernel(void* a, void* b, void* c, int n);
__global__ void scalarAddKernel(void*, float, void*, int);
__global__ void scalarMulKernel(void*, float, void*, int);
__global__ void matRelu(void* a, int n);
__global__ void matDRelu(void* a, int n);
__global__ void matTanh(void* a, int n);
__global__ void matDTanh(void* a, int n);
__global__ void init(unsigned int seed, curandState_t* states);
__global__ void randomizeKernel(curandState_t* states, float* a, int n);
__global__ void rowGetter(void* src, void* dest, size_t first, size_t last, size_t col);
__global__ void f2h(float*, void*, int n);
__global__ void h2f(void*, float*, int n);

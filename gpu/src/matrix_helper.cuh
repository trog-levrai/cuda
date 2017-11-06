# pragma once

# include <functional>
# include <curand.h>
# include <curand_kernel.h>

__global__ void vecMulKernel(float* a, float* b, float* c, int n);
__global__ void vecAddKernel(float* a, float* b, float* c, int n);
__global__ void vecSubKernel(float* a, float* b, float* c, int n);
__global__ void vecMulKernel(float* a, float* b, float* c, int n);
__global__ void scalarAddKernel(float*, float, float*, int);
__global__ void matTransformKernel(float* a, float (*f)(float), int n);
__global__ void matRelu(float* a, int n);
__global__ void matDRelu(float* a, int n);
__global__ void init(unsigned int seed, curandState_t* states);
__global__ void randomizeKernel(curandState_t* states, float* a, int n);
__global__ void rowGetter(float* src, float* dest, size_t first, size_t last, size_t col);

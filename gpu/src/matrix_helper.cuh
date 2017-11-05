# pragma once

# include <functional>

__global__ void vecMulKernel(float* a, float* b, float* c, int n);
__global__ void vecAddKernel(float* a, float* b, float* c, int n);
__global__ void vecSubKernel(float* a, float* b, float* c, int n);
__global__ void vecMulKernel(float* a, float* b, float* c, int n);
__global__ void scalarAddKernel(float*, float, float*, int);

__global__ void matTransformKernel(float* a, float (*f)(float), int n);

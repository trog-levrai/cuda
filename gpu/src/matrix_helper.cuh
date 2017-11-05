#pragma once

__global__ vecMulKernel(float* a, float* b, float* c, int n);
__global__ vecAddKernel(float* a, float* b, float* c, int n);
__global__ vecSubKernel(float* a, float* b, float* c, int n);
__global__ void vecMulKernel(float* a, float* b, float* c, int n);
__global__ void matTransformKernel(float* a, std::function<float (float)>, int n);

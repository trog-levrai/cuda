#pragma once

__global__ void vecMulKernel(float* a, float* b, float* c, int n);

__global__ void matTransformKernel(float* a, std::function<float (float)>, int n);

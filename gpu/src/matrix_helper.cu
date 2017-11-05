__global__ void vecMulKernel(float* A, float* B, float* C, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i<n) C[i] = A[i] * B[i];
}

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
  if (i<n) C[i] = A[i] + s;
}

__global__ void matTransformKernel(float* A, std::function<float (float)> f, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i<n) A[i] = f(A[i]);
}

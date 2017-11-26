# pragma once

# include <cuda.h>
# include "cublas_v2.h"

// Define this to turn on error checking
# define CUDA_ERROR_CHECK

# define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
# define CublasSafeCall( err ) __cublasSafeCall( err, __FILE__, __LINE__ )

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
# ifdef CUDA_ERROR_CHECK
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString(err));
        exit(-1);
    }
# endif

    return;
}

static const char* cublasGetErrorString(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line)
{
# ifdef CUDA_ERROR_CHECK
    if (CUBLAS_STATUS_SUCCESS != err)
    {
        fprintf(stderr, "cublasSafeCall() failed at %s:%i : %s\n",
                 file, line, cublasGetErrorString(err));
        exit(-1);
    }
# endif

    return;
}

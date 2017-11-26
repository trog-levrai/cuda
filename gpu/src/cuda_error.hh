# pragma once

# include <cuda.h>

// Define this to turn on error checking
# define CUDA_ERROR_CHECK

# define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
# ifdef CUDA_ERROR_CHECK
    if (cudaSuccess != err)
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
# endif

    return;
}

#define MAX_ORDER       9
#define DEGREE          13
#define DEGREE1         (DEGREE+1)
#define INTERVAL        2.5
#define INTERVALS       40

#include "gpu_compat.h"

#ifdef USE_SYCL
// The tables are `__device__` globals on CUDA; on SYCL they become constexpr
// so they can be read from any kernel without a device_global.
#pragma push_macro("__device__")
#undef __device__
#define __device__ inline constexpr
#include "rys_roots_dat.cu"
#pragma pop_macro("__device__")

#else
extern __device__ double ROOT_SMALLX_R0[];
extern __device__ double ROOT_SMALLX_R1[];
extern __device__ double ROOT_SMALLX_W0[];
extern __device__ double ROOT_SMALLX_W1[];
extern __device__ double ROOT_LARGEX_R_DATA[];
extern __device__ double ROOT_LARGEX_W_DATA[];
extern __device__ double ROOT_RW_DATA[];
#endif

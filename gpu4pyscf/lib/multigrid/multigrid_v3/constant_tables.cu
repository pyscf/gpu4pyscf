// CUDA-only constant tables (SYCL uses constexpr/device_global).
// Compiled only when NOT USE_SYCL (see multigrid_v3/CMakeLists.txt).
#include "constant_objects.cuh"

__constant__ int c_nf[] = {
    1,
    3,
    6,
    10,
    15,
    21,
    28,
    36,
    45,
};

__constant__ float c_div_nf[] = {
    1.f,
    0.333334f,
    0.166667f,
    0.100001f,
    0.066667f,
    0.047620f,
    0.035715f,
    0.027778f,
    0.022223f,
};

/* CUDA -> HIP compatibility stub. This directory is on the include path
 * only for HIP builds. The kernels use the CUDA-compatible x/y layout. */
#pragma once

#include <hip/hip_complex.h>

typedef hipDoubleComplex cuDoubleComplex;

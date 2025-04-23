#ifndef GPU4PYSCF_CONSTANT_CUH
#define GPU4PYSCF_CONSTANT_CUH

#include "gint/gint.h"

#ifdef USE_SYCL
#include "gint/sycl_device.hpp"

extern SYCL_EXTERNAL sycl_device_global<BasisProdCache> s_bpcache;
//extern SYCL_EXTERNAL sycl_device_global<int16_t[NFffff*3]> c_idx4c;
// extern SYCL_EXTERNAL sycl_device_global<int[TOT_NF*3]> s_idx;
// extern SYCL_EXTERNAL sycl_device_global<int[GPU_LMAX+2]> s_l_locs;
#else // USE_SYCL
extern __constant__ BasisProdCache c_bpcache;
//extern __constant__ int16_t c_idx4c[NFffff*3];
extern __constant__ int c_idx[TOT_NF*3];
extern __constant__ int c_l_locs[GPU_LMAX+2];
#endif // USE_SYCL

#endif //GPU4PYSCF_CONSTANT_CUH

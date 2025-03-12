#ifndef GPU4PYSCF_CONSTANT_HPP
#define GPU4PYSCF_CONSTANT_HPP

#include "gint/sycl_device.hpp"
#include "gint/gint.h"

extern SYCL_EXTERNAL sycl_device_global<BasisProdCache> c_bpcache;
extern SYCL_EXTERNAL sycl_device_global<int16_t[NFffff*3]> c_idx4c;
extern SYCL_EXTERNAL sycl_device_global<int[TOT_NF*3]> c_idx;
extern SYCL_EXTERNAL sycl_device_global<int[GPU_LMAX+2]> c_l_locs;

extern SYCL_EXTERNAL sycl_device_global<BasisProdOffsets[MAX_STREAMS]> c_offsets;
extern SYCL_EXTERNAL sycl_device_global<GINTEnvVars[MAX_STREAMS]> c_envs;
extern SYCL_EXTERNAL sycl_device_global<JKMatrix[MAX_STREAMS]> c_jk;

#endif //GPU4PYSCF_CONSTANT_HPP

#ifndef GPU4PYSCF_CONSTANT_CUH
#define GPU4PYSCF_CONSTANT_CUH

#include "gint/gint.h"

extern __constant__ BasisProdCache c_bpcache;
//extern __constant__ int16_t c_idx4c[NFffff*3];
extern __constant__ int c_idx[TOT_NF*3];
extern __constant__ int c_l_locs[GPU_LMAX+2];

#endif //GPU4PYSCF_CONSTANT_CUH

#ifndef GPU4PYSCF_CONSTANT_CUH
#define GPU4PYSCF_CONSTANT_CUH

#include "gint/gint.h"

#ifdef USE_SYCL
#include "gsycl/gpu_compat.h"

// Named distinctly from gint's s_gint_bpcache (gint/cint2e.cuh) -- both are
// GLOBAL DEFAULT-visibility device_global objects and libgint.so/libgvhf.so
// are co-resident in the process with no DT_NEEDED link between them, so
// identical names alias to whichever library's definition the dynamic
// linker resolves first, so gvhf's bpcache device_global is renamed to
// keep it distinct from gint's.
extern SYCL_EXTERNAL sycl_device_global<BasisProdCache> s_gvhf_bpcache;

#else // USE_SYCL
extern __constant__ BasisProdCache c_bpcache;
//extern __constant__ int16_t c_idx4c[NFffff*3];
extern __constant__ int c_idx[TOT_NF*3];
extern __constant__ int c_l_locs[GPU_LMAX+2];
#endif // USE_SYCL

#endif //GPU4PYSCF_CONSTANT_CUH

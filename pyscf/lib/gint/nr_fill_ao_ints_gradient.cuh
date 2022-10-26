#ifndef GPU4PYSCF_NR_FILL_AO_INTS_GRADIENT_CUH
#define GPU4PYSCF_NR_FILL_AO_INTS_GRADIENT_CUH


#include "gint.h"
#include "config.h"
#include "cuda_alloc.cuh"
#include "g2e.h"

#include "nr_fill_ao_ints.cuh"

typedef struct {
  size_t stride_xyz;

  int * primitive_function_offsets;
  double * exponents;
} GradientExtraInfo;

__host__
static int GINTfill_int2e_tasks(ERITensor *eri, BasisProdOffsets *offsets, GINTEnvVars *envs,
                                GradientExtraInfo * extra_info);

extern "C" {
__host__

void GINTdel_basis_prod(BasisProdCache ** pbp);

void GINTinit_gradient_extra_info(GradientExtraInfo ** gradient_extra_info,
                                  const int *bas, int nbas, const double *env,
                                  size_t stride_xyz);

void GINTdel_gradient_extra_info(GradientExtraInfo ** extra_info);

}
#endif //GPU4PYSCF_NR_FILL_AO_INTS_GRADIENT_CUH
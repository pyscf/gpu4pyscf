#ifndef GPU4PYSCF_NR_FILL_AO_INTS_GRADIENT_CUH
#define GPU4PYSCF_NR_FILL_AO_INTS_GRADIENT_CUH


#include "gint.h"
#include "config.h"
#include "cuda_alloc.cuh"
#include "g2e.h"

#include "nr_fill_ao_ints.cuh"

__host__
static int GINTfill_int2e_tasks(ERITensor *eri, BasisProdOffsets *offsets, GINTEnvVars *envs);

extern "C" {
__host__

void GINTdel_basis_prod(BasisProdCache ** pbp);

}
#endif //GPU4PYSCF_NR_FILL_AO_INTS_GRADIENT_CUH
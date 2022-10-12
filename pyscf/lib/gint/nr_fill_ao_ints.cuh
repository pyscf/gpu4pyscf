#ifndef GPU4PYSCF_NR_FILL_AO_INTS_CUH
#define GPU4PYSCF_NR_FILL_AO_INTS_CUH

#include "gint.h"
#include "config.h"
#include "cuda_alloc.cuh"
#include "g2e.h"

typedef struct {
  int stride_j;
  int stride_k;
  int stride_l;
  int stride_xyz;
  int ao_offsets_k;
  int ao_offsets_l;
  int nao;
  double *data;
} ERITensor;

#include "fill_ints.cu"
#include "g2e.cu"
#include "rys_roots.cu"
#include "g2e_root2.cu"
#include "g2e_root3.cu"

__host__
static int GINTfill_int2e_tasks(ERITensor *eri, BasisProdOffsets *offsets, GINTEnvVars *envs);

extern "C" {
__host__

void GINTdel_basis_prod(BasisProdCache ** pbp);

void GINTinit_basis_prod(BasisProdCache ** pbp, double diag_fac, int * ao_loc,
                         int * bas_pair2shls, int * bas_pairs_locs, int ncptype,
                         int * atm, int natm, int * bas, int nbas,
                         double * env);

int GINTfill_int2e(BasisProdCache * bpcache, double * eri, int nao,
                   int * strides, int * ao_offsets,
                   int * bins_locs_ij, int * bins_locs_kl, int nbins,
                   int cp_ij_id, int cp_kl_id);

}
#endif //GPU4PYSCF_NR_FILL_AO_INTS_CUH

#include <cassert>

#include "nr_fill_ao_ints_gradient.cuh"
#include "g2e_gradient.cu"
#include "cuda_alloc.cuh"

#define print printf("The code has arrived here\n");

void GINTinit_uw_s1_nabla1i(double *uw_buf, BasisProdOffsets *offsets,
                            GINTEnvVars *envs, BasisProdCache *bpcache)
{
  size_t ntasks_ij = offsets->ntasks_ij;
  size_t ntasks_kl = offsets->ntasks_kl;
  int nprim_ij = envs->nprim_ij;
  int nprim_kl = envs->nprim_kl;
  int nroots = envs->nrys_roots + 1;
  int strides = envs->nprim_ij * envs->nprim_kl * nroots * 2;
  int n_primitive_pairs = bpcache->primitive_pairs_locs[bpcache->ncptype];
  double *a12 = bpcache->aexyz;
  double *x12 = bpcache->aexyz + n_primitive_pairs * 2;
  double *y12 = bpcache->aexyz + n_primitive_pairs * 3;
  double *z12 = bpcache->aexyz + n_primitive_pairs * 4;

#pragma omp parallel
  {
    int ij, kl, task_ij, task_kl, prim_ij, prim_kl;
    size_t n;
    double *uw;
#pragma omp for schedule(static)
    for (n = 0; n < ntasks_ij*ntasks_kl; n++) {
      task_ij = n % ntasks_ij;
      task_kl = n / ntasks_ij;
      prim_ij = offsets->primitive_ij + task_ij * nprim_ij;
      prim_kl = offsets->primitive_kl + task_kl * nprim_kl;
      uw = uw_buf + n * strides;
      for (ij = prim_ij; ij < prim_ij+nprim_ij; ij++) {
        for (kl = prim_kl; kl < prim_kl+nprim_kl; kl++) {
          double aij = a12[ij];
          double xij = x12[ij];
          double yij = y12[ij];
          double zij = z12[ij];
          double akl = a12[kl];
          double xkl = x12[kl];
          double ykl = y12[kl];
          double zkl = z12[kl];
          double rx = xij - xkl;
          double ry = yij - ykl;
          double rz = zij - zkl;
          double aijkl = aij + akl;
          double a0 = aij * akl / aijkl;
          double x = a0 * (rx * rx + ry * ry + rz * rz);
          double *u = uw;
          double *w = uw + nroots;
          CINTrys_roots(nroots, x, u, w);
          uw += nroots * 2;
        } }
    }
  }

__host__
static int GINTfill_nabla1i_int2e_tasks(ERITensor *eri,
                                        BasisProdOffsets *offsets,
                                        GINTEnvVars *envs)
{
  int nrys_roots = envs->nrys_roots;
  int ntasks_ij = offsets->ntasks_ij;
  int ntasks_kl = offsets->ntasks_kl;
  assert(ntasks_kl < 65536*THREADSY);
  int type_ijkl;

  dim3 threads(THREADSX, THREADSY);
  dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ntasks_kl+THREADSY-1)/THREADSY);
  switch (nrys_roots) {
    case 1:
      type_ijkl = (envs->i_l << 3) | (envs->j_l << 2) | (envs->k_l << 1) | envs->l_l;
      //GINTfill_int2e_kernel<1, GOUTSIZE1> <<<blocks, threads>>>(*offsets);
      switch (type_ijkl) {
        case 0b0000: GINTfill_nabla1i_int2e_kernel0000<<<blocks, threads>>>(*eri, *offsets); break;
        default:
          fprintf(stderr, "troots=1 ype_ijkl %d\n", type_ijkl);
      }
      break;

    default:
      fprintf(stderr, "rys roots %d\n", nrys_roots);
      return 1;
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Error of GINTfill_int2e_kernel: %s\n", cudaGetErrorString(err));
    return 1;
  }
  return 0;
}

extern "C" {__host__

int GINTfill_nabla1i_int2e(BasisProdCache *bpcache,
                           double *eri, int nao,
                           int *strides, int *ao_offsets,
                           int *bins_locs_ij, int *bins_locs_kl, int nbins,
                           int cp_ij_id, int cp_kl_id)
{
  ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
  ContractionProdType *cp_kl = bpcache->cptype + cp_kl_id;
  GINTEnvVars envs;
  GINTinit_EnvVars(&envs, cp_ij, cp_kl);
  if (envs.nrys_roots >= 8) {
    return 2;
  }

  if (envs.nrys_roots > 2) {
    int16_t *idx4c = (int16_t *)malloc(sizeof(int16_t) * envs.nf * 3);
    int *idx_ij = (int *)malloc(sizeof(int) * envs.nfi * envs.nfj * 3);
    int *idx_kl = (int *)malloc(sizeof(int) * envs.nfk * envs.nfl * 3);
    GINTinit_2c_gidx(idx_ij, cp_ij->l_bra, cp_ij->l_ket);
    GINTinit_2c_gidx(idx_kl, cp_kl->l_bra, cp_kl->l_ket);
    GINTinit_4c_idx(idx4c, idx_ij, idx_kl, &envs);
    if (envs.nf > NFffff) {
      DEVICE_INIT(int16_t, d_idx4c, idx4c, envs.nf * 3);
      envs.idx = d_idx4c;
    } else {
      checkCudaErrors(cudaMemcpyToSymbol(c_idx4c, idx4c, sizeof(int16_t)*envs.nf*3));
    }
    free(idx4c);
    free(idx_ij);
    free(idx_kl);
  }

  int nrys_roots_gradient = envs.nrys_roots + 1;

  // Data and buffers to be allocated on-device. Allocate them here to
  // reduce the calls to malloc
  int nroots2 = nrys_roots_gradient * 2;
  int kl_bin, ij_bin1;
  double *uw_buf, *d_uw;
  size_t uw_size = 0;
  if (nrys_roots_gradient + 1 > POLYFIT_ORDER) {
    for (kl_bin = 0; kl_bin < nbins; ++kl_bin) {
      ij_bin1 = nbins - kl_bin;
      int bas_ij0 = bins_locs_ij[0];
      int bas_ij1 = bins_locs_ij[ij_bin1];
      int bas_kl0 = bins_locs_kl[kl_bin];
      int bas_kl1 = bins_locs_kl[kl_bin+1];
      int ntasks_ij = bas_ij1 - bas_ij0;
      int ntasks_kl = bas_kl1 - bas_kl0;
      uw_size = MAX(uw_size, ntasks_ij * ntasks_kl);
    }
    uw_size *= envs.nprim_ij * envs.nprim_kl * nroots2;
    checkCudaErrors(cudaHostAlloc(&uw_buf, sizeof(double) * uw_size,
                                  cudaHostAllocMapped));
    checkCudaErrors(cudaMalloc(&d_uw, sizeof(double) * uw_size));
    envs.uw = d_uw;
  }
  checkCudaErrors(cudaMemcpyToSymbol(c_envs, &envs, sizeof(GINTEnvVars)));
  // move bpcache to constant memory
  checkCudaErrors(cudaMemcpyToSymbol(c_bpcache, bpcache, sizeof(BasisProdCache)));

  ERITensor eritensor;
  eritensor.stride_j = strides[1];
  eritensor.stride_k = strides[2];
  eritensor.stride_l = strides[3];
  eritensor.n_elem = nao * strides[3];
  eritensor.ao_offsets_k = ao_offsets[2];
  eritensor.ao_offsets_l = ao_offsets[3];
  eritensor.nao = nao;
  eritensor.data = eri;

  BasisProdOffsets offsets;
  int *bas_pairs_locs = bpcache->bas_pairs_locs;
  int *primitive_pairs_locs = bpcache->primitive_pairs_locs;
  for (kl_bin = 0; kl_bin < nbins; kl_bin++) {
    int bas_kl0 = bins_locs_kl[kl_bin];
    int bas_kl1 = bins_locs_kl[kl_bin+1];
    int ntasks_kl = bas_kl1 - bas_kl0;
    if (ntasks_kl <= 0) {
      continue;
    }
    // ij_bin + kl_bin < nbins <~> e_ij*e_kl < cutoff
    ij_bin1 = nbins - kl_bin;
    int bas_ij0 = bins_locs_ij[0];
    int bas_ij1 = bins_locs_ij[ij_bin1];
    int ntasks_ij = bas_ij1 - bas_ij0;
    if (ntasks_ij <= 0) {
      continue;
    }
    offsets.ntasks_ij = ntasks_ij;
    offsets.ntasks_kl = ntasks_kl;
    offsets.bas_ij = bas_pairs_locs[cp_ij_id] + bas_ij0;
    offsets.bas_kl = bas_pairs_locs[cp_kl_id] + bas_kl0;
    offsets.primitive_ij = primitive_pairs_locs[cp_ij_id] + bas_ij0 * envs.nprim_ij;
    offsets.primitive_kl = primitive_pairs_locs[cp_kl_id] + bas_kl0 * envs.nprim_kl;

    if (nrys_roots_gradient > POLYFIT_ORDER) {
      // move rys roots and weights to device
      GINTinit_uw_s1(uw_buf, &offsets, &envs, bpcache);
      uw_size = (size_t)ntasks_ij * ntasks_kl * envs.nprim_ij * envs.nprim_kl * nroots2;
      checkCudaErrors(cudaMemcpy(d_uw, uw_buf, sizeof(double) * uw_size,
                                 cudaMemcpyHostToDevice));
    }
    int err = GINTfill_nabla1i_int2e_tasks(&eritensor, &offsets, &envs);
    if (err != 0) {
      return err;
    }
  }

  if (nrys_roots_gradient > POLYFIT_ORDER) {
    checkCudaErrors(cudaFreeHost(uw_buf));
    FREE(d_uw);
  }
  if (envs.nf > NFffff) {
    FREE(envs.idx);
  }
  return 0;

}
}

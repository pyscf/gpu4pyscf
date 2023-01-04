#include <cassert>

#include "nr_fill_ao_ints_gradient.cuh"
#include "g2e_gradient.cu"
#include "g2e_root2_gradient.cu"
#include "g2e_root3_gradient.cu"
#include "cuda_alloc.cuh"
#include "rys_roots.cuh"

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
          fprintf(stderr, "roots=1 type_ijkl %d\n", type_ijkl);
      }
      break;

    case 2:
      type_ijkl = (envs->i_l << 6) | (envs->j_l << 4) | (envs->k_l << 2) | envs->l_l;
      switch (type_ijkl) {
        case (0<<6)|(0<<4)|(1<<2)|0: GINTfill_int2e_kernel_nabla1i_0010<<<blocks, threads>>>(*eri, *offsets); break;
        case (0<<6)|(0<<4)|(1<<2)|1: GINTfill_int2e_kernel_nabla1i_0011<<<blocks, threads>>>(*eri, *offsets); break;
        case (0<<6)|(0<<4)|(2<<2)|0: GINTfill_int2e_kernel_nabla1i_0020<<<blocks, threads>>>(*eri, *offsets); break;
        case (1<<6)|(0<<4)|(0<<2)|0: GINTfill_int2e_kernel_nabla1i_1000<<<blocks, threads>>>(*eri, *offsets); break;
        case (1<<6)|(0<<4)|(1<<2)|0: GINTfill_int2e_kernel_nabla1i_1010<<<blocks, threads>>>(*eri, *offsets); break;
        case (1<<6)|(1<<4)|(0<<2)|0: GINTfill_int2e_kernel_nabla1i_1100<<<blocks, threads>>>(*eri, *offsets); break;
        case (2<<6)|(0<<4)|(0<<2)|0: GINTfill_int2e_kernel_nabla1i_2000<<<blocks, threads>>>(*eri, *offsets); break;
        default:
          fprintf(stderr, "roots=2 type_ijkl %d\n", type_ijkl);
      }
      break;

    case 3:
      type_ijkl = (envs->i_l << 6) | (envs->j_l << 4) | (envs->k_l << 2) | envs->l_l;
      switch (type_ijkl) {
        case (0<<6)|(0<<4)|(2<<2)|1: GINTfill_int2e_kernel_nabla1i_0021<<<blocks, threads>>>(*eri, *offsets); break;
        case (0<<6)|(0<<4)|(2<<2)|2: GINTfill_int2e_kernel_nabla1i_0022<<<blocks, threads>>>(*eri, *offsets); break;
        case (0<<6)|(0<<4)|(3<<2)|0: GINTfill_int2e_kernel_nabla1i_0030<<<blocks, threads>>>(*eri, *offsets); break;
        case (0<<6)|(0<<4)|(3<<2)|1: GINTfill_int2e_kernel_nabla1i_0031<<<blocks, threads>>>(*eri, *offsets); break;
        case (1<<6)|(0<<4)|(1<<2)|1: GINTfill_int2e_kernel_nabla1i_1011<<<blocks, threads>>>(*eri, *offsets); break;
        case (1<<6)|(0<<4)|(2<<2)|0: GINTfill_int2e_kernel_nabla1i_1020<<<blocks, threads>>>(*eri, *offsets); break;
        case (1<<6)|(0<<4)|(2<<2)|1: GINTfill_int2e_kernel_nabla1i_1021<<<blocks, threads>>>(*eri, *offsets); break;
        case (1<<6)|(0<<4)|(3<<2)|0: GINTfill_int2e_kernel_nabla1i_1030<<<blocks, threads>>>(*eri, *offsets); break;
        case (1<<6)|(1<<4)|(1<<2)|0: GINTfill_int2e_kernel_nabla1i_1110<<<blocks, threads>>>(*eri, *offsets); break;
        case (1<<6)|(1<<4)|(1<<2)|1: GINTfill_int2e_kernel_nabla1i_1111<<<blocks, threads>>>(*eri, *offsets); break;
        case (1<<6)|(1<<4)|(2<<2)|0: GINTfill_int2e_kernel_nabla1i_1120<<<blocks, threads>>>(*eri, *offsets); break;
        case (2<<6)|(0<<4)|(1<<2)|0: GINTfill_int2e_kernel_nabla1i_2010<<<blocks, threads>>>(*eri, *offsets); break;
        case (2<<6)|(0<<4)|(1<<2)|1: GINTfill_int2e_kernel_nabla1i_2011<<<blocks, threads>>>(*eri, *offsets); break;
        case (2<<6)|(0<<4)|(2<<2)|0: GINTfill_int2e_kernel_nabla1i_2020<<<blocks, threads>>>(*eri, *offsets); break;
        case (2<<6)|(1<<4)|(0<<2)|0: GINTfill_int2e_kernel_nabla1i_2100<<<blocks, threads>>>(*eri, *offsets); break;
        case (2<<6)|(1<<4)|(1<<2)|0: GINTfill_int2e_kernel_nabla1i_2110<<<blocks, threads>>>(*eri, *offsets); break;
        case (2<<6)|(2<<4)|(0<<2)|0: GINTfill_int2e_kernel_nabla1i_2200<<<blocks, threads>>>(*eri, *offsets); break;
        case (3<<6)|(0<<4)|(0<<2)|0: GINTfill_int2e_kernel_nabla1i_3000<<<blocks, threads>>>(*eri, *offsets); break;
        case (3<<6)|(0<<4)|(1<<2)|0: GINTfill_int2e_kernel_nabla1i_3010<<<blocks, threads>>>(*eri, *offsets); break;
        case (3<<6)|(1<<4)|(0<<2)|0: GINTfill_int2e_kernel_nabla1i_3100<<<blocks, threads>>>(*eri, *offsets); break;
        default:
          fprintf(stderr, "roots=3 type_ijkl %d\n", type_ijkl);
      }
      break;

    case 4:
      GINTfill_int2e_kernel_nabla1i<4, NABLAGSIZE4> <<<blocks, threads>>>(*eri, *offsets);
      break;
    case 5:
      GINTfill_int2e_kernel_nabla1i<5, NABLAGSIZE5> <<<blocks, threads>>>(*eri, *offsets);
      break;
    case 6:
      GINTfill_int2e_kernel_nabla1i<6, NABLAGSIZE6> <<<blocks, threads>>>(*eri, *offsets);
      break;
    case 7:
      GINTfill_int2e_kernel_nabla1i<7, NABLAGSIZE7> <<<blocks, threads>>>(*eri, *offsets);
      break;
      //case 8:
      //    GINTfill_int2e_kernel<8, GOUTSIZE8> <<<blocks, threads>>>(*offsets);
      //    break;
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
  GINTinit_EnvVars_nabla1i(&envs, cp_ij, cp_kl);
  if (envs.nrys_roots >= 8) {
    return 2;
  }

  if (envs.nrys_roots > 2) {
    int16_t *idx4c = (int16_t *)malloc(sizeof(int16_t) * envs.nf * 3);
    int *idx_ij = (int *)malloc(sizeof(int) * envs.nfi * envs.nfj * 3);
    int *idx_kl = (int *)malloc(sizeof(int) * envs.nfk * envs.nfl * 3);
    GINTinit_2c_gidx_nabla1i(idx_ij, cp_ij->l_bra, cp_ij->l_ket);
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

  // Data and buffers to be allocated on-device. Allocate them here to
  // reduce the calls to malloc
  int nroots2 = envs.nrys_roots * 2;
  int kl_bin, ij_bin1;
  double *uw_buf, *d_uw;
  size_t uw_size = 0;
  if (envs.nrys_roots > POLYFIT_ORDER) {
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

    if (envs.nrys_roots > POLYFIT_ORDER) {
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

  if (envs.nrys_roots > POLYFIT_ORDER) {
    checkCudaErrors(cudaFreeHost(uw_buf));
    FREE(d_uw);
  }
  if (envs.nf > NFffff) {
    FREE(envs.idx);
  }
  return 0;

}
}

#include "nr_fill_ao_ints_gradient.cuh"
#include "g2e_gradient.cu"
__host__
static int GINTfill_int2e_tasks(ERITensor *eri, BasisProdOffsets *offsets, GINTEnvVars *envs,
                                GradientExtraInfo * extra_info)
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
        case 0b0000: GINTfill_nabla1i_int2e_kernel0000<<<blocks, threads>>>(*eri, *offsets, extra_info); break;
        case 0b0010: GINTfill_int2e_kernel0010<<<blocks, threads>>>(*eri, *offsets); break;

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

void GINTinit_gradient_extra_info(GradientExtraInfo * gradient_extra_info,
                         const int *bas, const int nbas, const double *env,
                         const size_t stride_xyz)
{
  GradientExtraInfo *info = (GradientExtraInfo *)malloc(sizeof(GradientExtraInfo));
  memset(info, 0, sizeof(GradientExtraInfo));

  double * exponents = (double *) malloc(nbas * sizeof(double));
  for(int basis_id = 0; basis_id < nbas; basis_id++) {
    exponents[basis_id] = bas[NPRIM_OF + basis_id * BAS_SLOTS];
  }
  // initialize pair data on GPU memory
  DEVICE_INIT(double, d_exponents, exponents, nbas * sizeof(double));
  info->exponents = d_exponents;
  info->stride_xyz = stride_xyz;

  free(exponents);
}

void GINTdel_gradient_extra_info(GradientExtraInfo * extra_info) {
  FREE(extra_info->exponents);
  free(extra_info);
}


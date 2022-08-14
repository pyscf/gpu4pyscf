#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cint.h>
#include "gint/cuda_alloc.cuh"
#include "nr_eval_gto.cuh"

#define THREADSX        32
#define THREADSY        4
#define THREADSXY       (THREADSX * THREADSY)
#define THREADSYY       (THREADSY * THREADSY)
#define DIVXY           (THREADSX / THREADSY)

__global__
static void _dot_ao_dm(double *out, double *ao, double *dm,
                       int jsh0, int jsh1, int ngrids, int nbins, int nsegs,
                       int *bas_segs, uint8_t *screen_index, uint8_t *pair_mask)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int grid_blk = blockIdx.x;
    int shell_blk = blockIdx.y;
    int jsh = jsh0 + shell_blk * THREADSY + ty;
    if (jsh >= jsh1) {
        return;
    }

    int nbas = c_envs.nbas;
    int bas_blocks = (nbas + THREADSY - 1) / THREADSY;
    int jsh_blk = jsh0 / THREADSY + shell_blk;
    uint8_t sj = screen_index[grid_blk * bas_blocks + jsh_blk];
    if (sj == 0) {
        return;
    }

    uint8_t nbins_i;
    if (nbins > sj) {
        nbins_i = nbins - sj;
    } else {
        nbins_i = 1;
    }

    int *ao_loc = c_envs.ao_loc;
    int grid_id = grid_blk * THREADSX + tx;
    int jp = blockIdx.z;
    int j = ao_loc[jsh] + jp;
    int ish, ip, n, k, i, seg;
    size_t Nao = ao_loc[nbas];
    size_t Ngrids = ngrids;
    double val = 0;

    __shared__ double s_ao[THREADSX*THREADSY];
    __shared__ double s_dm[THREADSX*THREADSY];

    for (seg = 0; seg < nsegs; seg++) {
        int ish0 = bas_segs[seg];
        int ish1 = bas_segs[seg+1];
        int nsh = ish1 - ish0;
        int degen = ao_loc[ish0+1] - ao_loc[ish0];
        int i0 = ao_loc[ish0];
        for (ish = 0; ish < nsh; ish+=THREADSX) {
            for (ip = 0; ip < degen; ip++) {
                i = i0 + ish * degen + ip;
                s_dm[ty*THREADSX+tx] = dm[i * Nao + j];

                for (n = 0; n < THREADSX; n+=THREADSY) {
                    int ishp = ish0 + ish + n;
                    int ish_blk = ishp / THREADSY;
                    if (ishp < ish1 &&
                        screen_index[grid_blk * bas_blocks + ish_blk] > nbins_i &&
                        pair_mask[ish_blk * bas_blocks + jsh_blk]) {
                        i = i0 + ishp * degen + ip;
                        s_ao[ty*THREADSX+tx] = ao[i*Ngrids+grid_id];
                        __syncthreads();
                        if (ishp + THREADSY < ish1) {
                            for (k = 0; k < THREADSY; k++) {
                                val += s_ao[k*THREADSX+tx] * s_dm[ty*THREADSX+n+k];
                            }
                        } else {
                            for (k = 0; k < ish1 - ishp; k++) {
                                val += s_ao[k*THREADSX+tx] * s_dm[ty*THREADSX+n+k];
                            }
                        }
                        __syncthreads();
                    }
                }
            }
        }
    }

    if (grid_id < ngrids) {
        ao[j*Ngrids+grid_id] += val;
    }
}

__global__
static void _dot_ao_dmT(double *out, double *ao, double *dm,
                        int jsh0, int jsh1, int ngrids, int nbins, int nsegs,
                        int *bas_segs, uint8_t *screen_index, uint8_t *pair_mask)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int grid_blk = blockIdx.x;
    int shell_blk = blockIdx.y;
    int jsh = jsh0 + shell_blk * THREADSY + ty;
    if (jsh >= jsh1) {
        return;
    }

    int nbas = c_envs.nbas;
    int bas_blocks = (nbas + THREADSY - 1) / THREADSY;
    int jsh_blk = jsh0 / THREADSY + shell_blk;
    uint8_t sj = screen_index[grid_blk * bas_blocks + jsh_blk];
    if (sj == 0) {
        return;
    }

    uint8_t nbins_i;
    if (nbins > sj) {
        nbins_i = nbins - sj;
    } else {
        nbins_i = 1;
    }

    int *ao_loc = c_envs.ao_loc;
    int grid_id = grid_blk * THREADSX + tx;
    int jp = blockIdx.z;
    int j = ao_loc[jsh] + jp;
    int ish, ip, n, k, i, seg;
    size_t Nao = ao_loc[nbas];
    size_t Ngrids = ngrids;
    double val = 0;

    __shared__ double s_ao[THREADSX*THREADSY];
    __shared__ double s_dm[THREADSX*THREADSY];

    for (seg = 0; seg < nsegs; seg++) {
        int ish0 = bas_segs[seg];
        int ish1 = bas_segs[seg+1];
        int nsh = ish1 - ish0;
        int degen = ao_loc[ish0+1] - ao_loc[ish0];
        int i0 = ao_loc[ish0];
        for (ish = 0; ish < nsh; ish+=THREADSX) {
            for (ip = 0; ip < degen; ip++) {
                i = i0 + ish * degen + ip;
                s_dm[ty*THREADSX+tx] = dm[j * Nao + i];

                for (n = 0; n < THREADSX; n+=THREADSY) {
                    int ishp = ish0 + ish + n;
                    int ish_blk = ishp / THREADSY;
                    if (ishp < ish1 &&
                        screen_index[grid_blk * bas_blocks + ish_blk] > nbins_i &&
                        pair_mask[ish_blk * bas_blocks + jsh_blk]) {
                        i = i0 + ishp * degen + ip;
                        s_ao[ty*THREADSX+tx] = ao[i*Ngrids+grid_id];
                        __syncthreads();
                        if (ishp + THREADSY < ish1) {
                            for (k = 0; k < THREADSY; k++) {
                                val += s_ao[k*THREADSX+tx] * s_dm[ty*THREADSX+n+k];
                            }
                        } else {
                            for (k = 0; k < ish1 - ishp; k++) {
                                val += s_ao[k*THREADSX+tx] * s_dm[ty*THREADSX+n+k];
                            }
                        }
                        __syncthreads();
                    }
                }
            }
        }
    }

    if (grid_id < ngrids) {
        ao[j*Ngrids+grid_id] += val;
    }
}

__global__
static void _dot_aow_ao(double *out, double *bra, double *ket, double *wv,
                        int ngrids, int nbins, uint8_t *screen_index,
                        int *bas_pair2bra, int *bas_pair2ket)
{
    int task_ij = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int txy = ty * DIVXY + tx;
    int tyz = tz * THREADSY + ty;
    int ish0 = bas_pair2bra[task_ij];
    int jsh0 = bas_pair2ket[task_ij];
    int *ao_loc = c_envs.ao_loc;
    int i0 = ao_loc[ish0];
    int j0 = ao_loc[jsh0];
    int ish4 = ish0 / THREADSY;
    int jsh4 = jsh0 / THREADSY;
    int degen_i = gridDim.y;
    int degen_j = gridDim.z;
    int ip = blockIdx.y;
    int jp = blockIdx.z;

    int nbas = c_envs.nbas;
    int bas_blocks = (nbas + THREADSY - 1) / THREADSY;
    size_t Nao = ao_loc[nbas];
    size_t Ngrids = ngrids;
    double val = 0;

    __shared__ double s_bra[THREADSXY];
    __shared__ double s_ket[THREADSXY];

    int grid_blk;
    for (grid_blk = 0; grid_blk < ngrids/THREADSX; grid_blk++) {
        int grid0 = grid_blk * THREADSX;
        uint8_t si = screen_index[grid_blk*bas_blocks+ish4];
        uint8_t sj = screen_index[grid_blk*bas_blocks+jsh4];
        if (si + sj >= nbins) {
            int grid_id = grid0 + txy;
            for (int n = 0; n < THREADSY; n++) {
                int i = i0 + n * degen_i + ip;
                int j = j0 + n * degen_j + jp;
                s_bra[n*THREADSX+txy] = bra[i*Ngrids+grid_id];
                s_ket[n*THREADSX+txy] = ket[j*Ngrids+grid_id] * wv[grid_id];
            }
            __syncthreads();
            val += s_bra[ty*THREADSX+       +tx] * s_ket[tz*THREADSX+       +tx];
            val += s_bra[ty*THREADSX+DIVXY  +tx] * s_ket[tz*THREADSX+DIVXY  +tx];
            val += s_bra[ty*THREADSX+DIVXY*2+tx] * s_ket[tz*THREADSX+DIVXY*2+tx];
            val += s_bra[ty*THREADSX+DIVXY*3+tx] * s_ket[tz*THREADSX+DIVXY*3+tx];
            __syncthreads();
        }
    }
    int grid0 = grid_blk * THREADSX;
    if (grid0 < ngrids) {
        int grid_id = grid0 + txy;
        for (int n = 0; n < THREADSY; n++) {
            int i = i0 + n * degen_i + ip;
            int j = j0 + n * degen_j + jp;
            s_bra[n*THREADSX+txy] = bra[i*Ngrids+grid_id];
            s_ket[n*THREADSX+txy] = ket[j*Ngrids+grid_id] * wv[grid_id];
        }
        int bgrids = ngrids - grid0;
        __syncthreads();
        for (int n = 0; n < bgrids; n+=DIVXY) {
            if (n + tx < bgrids) {
                val += s_bra[ty*THREADSX+n+tx] * s_ket[tz*THREADSX+n+tx];
            }
        }
        __syncthreads();
    }

    double *val_buf = s_bra;
    val_buf[tx * THREADSYY + tyz] = val;
    __syncthreads();
    for (int n = (DIVXY>>1); n > 0; n >>= 1) {
        if (tx < n) {
            val_buf[tx*THREADSYY+tyz] += val_buf[(tx+n)*THREADSYY+tyz];
        }
        __syncthreads();
    }
    int i = i0 + ty * degen_i + ip;
    int j = j0 + tz * degen_j + jp;
    if (tx == 0) {
        out[i*Nao+j] = val_buf[tyz];
    }
}

__global__
static void _dot_ao_ao(double *out, double *bra, double *ket,
                       int ngrids, int nbins, uint8_t *screen_index,
                       int *bas_pair2bra, int *bas_pair2ket)
{
    int task_ij = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int txy = ty * DIVXY + tx;
    int tyz = tz * THREADSY + ty;
    int ish0 = bas_pair2bra[task_ij];
    int jsh0 = bas_pair2ket[task_ij];
    int *ao_loc = c_envs.ao_loc;
    int i0 = ao_loc[ish0];
    int j0 = ao_loc[jsh0];
    int ish4 = ish0 / THREADSY;
    int jsh4 = jsh0 / THREADSY;
    int degen_i = gridDim.y;
    int degen_j = gridDim.z;
    int ip = blockIdx.y;
    int jp = blockIdx.z;

    int nbas = c_envs.nbas;
    int bas_blocks = (nbas + THREADSY - 1) / THREADSY;
    size_t Nao = ao_loc[nbas];
    size_t Ngrids = ngrids;
    double val = 0;

    __shared__ double s_bra[THREADSXY];
    __shared__ double s_ket[THREADSXY];

    int grid_blk;
    for (grid_blk = 0; grid_blk < ngrids/THREADSX; grid_blk++) {
        int grid0 = grid_blk * THREADSX;
        uint8_t si = screen_index[grid_blk*bas_blocks+ish4];
        uint8_t sj = screen_index[grid_blk*bas_blocks+jsh4];
        if (si + sj >= nbins) {
            int grid_id = grid0 + txy;
            for (int n = 0; n < THREADSY; n++) {
                int i = i0 + n * degen_i + ip;
                int j = j0 + n * degen_j + jp;
                s_bra[n*THREADSX+txy] = bra[i*Ngrids+grid_id];
                s_ket[n*THREADSX+txy] = ket[j*Ngrids+grid_id];
            }
            __syncthreads();
            val += s_bra[ty*THREADSX+       +tx] * s_ket[tz*THREADSX+       +tx];
            val += s_bra[ty*THREADSX+DIVXY  +tx] * s_ket[tz*THREADSX+DIVXY  +tx];
            val += s_bra[ty*THREADSX+DIVXY*2+tx] * s_ket[tz*THREADSX+DIVXY*2+tx];
            val += s_bra[ty*THREADSX+DIVXY*3+tx] * s_ket[tz*THREADSX+DIVXY*3+tx];
            __syncthreads();
        }
    }
    int grid0 = grid_blk * THREADSX;
    if (grid0 < ngrids) {
        int grid_id = grid0 + txy;
        for (int n = 0; n < THREADSY; n++) {
            int i = i0 + n * degen_i + ip;
            int j = j0 + n * degen_j + jp;
            s_bra[n*THREADSX+txy] = bra[i*Ngrids+grid_id];
            s_ket[n*THREADSX+txy] = ket[j*Ngrids+grid_id];
        }
        int bgrids = ngrids - grid0;
        __syncthreads();
        for (int n = 0; n < bgrids; n+=DIVXY) {
            if (n + tx < bgrids) {
                val += s_bra[ty*THREADSX+n+tx] * s_ket[tz*THREADSX+n+tx];
            }
        }
        __syncthreads();
    }

    double *val_buf = s_bra;
    val_buf[tx * THREADSYY + tyz] = val;
    __syncthreads();
    for (int n = (DIVXY>>1); n > 0; n >>= 1) {
        if (tx < n) {
            val_buf[tx*THREADSYY+tyz] += val_buf[(tx+n)*THREADSYY+tyz];
        }
        __syncthreads();
    }
    int i = i0 + ty * degen_i + ip;
    int j = j0 + tz * degen_j + jp;
    if (tx == 0) {
        out[i*Nao+j] = val_buf[tyz];
    }
}


//// 'ip,ip->p'
//__global__
//static void _dcontract_rho_sparse(double *rho, double *bra, double *ket,
//                             int nao, int ngrids, int nbas,
//                             uint8_t *screen_index)
//{
//    int *ao_loc = c_envs.ao_loc;
//    int grid_blk = blockIdx.x;
//    int grid_id = grid_blk * THREADSX + threadIdx.x;
//
//    double val = 0;
//    for (ish = 0; ish < nbas; ish+=4) {
//        if (screen_index[grid_blk * nbas + ish]) {
//            i0 = ao_loc[ish];
//            i1 = ao_loc[ish+1];
//            for (i = i0; i < i1; i++) {
//                i_addr = i * Ngrids + ig0;
//                val += bra[i_addr+n] * ket[i_addr+n];
//            }
//        }
//    }
//    if (grid_id < ngrids) {
//        rho[grid_id] = val;
//    }
//}

//// 'nip,np->ip'
//__global__
//static void _dscale_ao_sparse(double *aow, double *ao, double *wv,
//                              int comp, int nao, int ngrids, int nbas,
//                              uint8_t *screen_index, int *ao_loc)
//{
//    int tx = threadIdx.x;
//    int ty = threadIdx.y;
//    int degen = gridDim.z;
//    int jp = blockIdx.z;
//    int grid_blk = blockIdx.x;
//    int shell_blk = blockIdx.y;
//    int grid_id = grid_blk * THREADSX + tx;
//    int shell_id = shell_blk * THREADSY + ty;
//    int i = ao_loc[shell_off+shell_id] + jp;
//    if (shell_id >= jsh1) {
//        return;
//    }
//
//    double val = 0;
//    for (ic = 0; ic < comp; ic++) {
//        val += ao[(ic * nao + i) * Ngrids + grid_id] * wv[ic * Ngrids + grid_id];
//    }
//
//    if (grid_id < ngrids) {
//        ao[jao*Ngrids+grid_id] += val;
//    }
//}

extern "C" {
__host__
int GDFTdot_ao_dm_sparse(double *out, double *ao, double *dm, int trans_dm,
                         int ngrids, int nbas, int nbins, int nsegs, int *seg_loc,
                         uint8_t *screen_index, uint8_t *pair_mask, int *ao_loc)
{
    int err_code = 0;
    int grid_blocks = (ngrids + THREADSX - 1) / THREADSX;
    int bas_blocks = (nbas + THREADSY - 1) / THREADSY;
    int nao = ao_loc[nbas];
    checkCudaErrors(cudaMemset(out, 0, sizeof(double)*ngrids*nao));
    DEVICE_INIT(uint8_t, d_sindex, screen_index, grid_blocks * bas_blocks);
    DEVICE_INIT(uint8_t, d_pair_mask, pair_mask, grid_blocks * bas_blocks);
    DEVICE_INIT(int, d_seg_loc, seg_loc, nsegs);

    for (int seg = 0; seg < nsegs; seg++) {
        int ish0 = seg_loc[seg];
        int ish1 = seg_loc[seg+1];
        // segments should be aligned to 4
        assert(ish1 % THREADSY == 0);
        int degen = ao_loc[ish0+1] - ao_loc[ish0];
        int nsh = ish1 - ish0;
        dim3 threads(THREADSX, THREADSY);
        dim3 blocks((ngrids+THREADSX-1)/THREADSX, (nsh+THREADSY-1)/THREADSY, degen);
        if (trans_dm) {
            _dot_ao_dmT<<<blocks, threads>>>(out, ao, dm, ish0, ish1, ngrids,
                                             nbins, nsegs, d_seg_loc, d_sindex,
                                             d_pair_mask);
        } else {
            _dot_ao_dm<<<blocks, threads>>>(out, ao, dm, ish0, ish1, ngrids,
                                            nbins, nsegs, d_seg_loc, d_sindex,
                                            d_pair_mask);
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error of GDFTdot_ao_dm_sparse: %s\n",
                    cudaGetErrorString(err));
            err_code = 1;
            goto cleanup;
        }
    }
cleanup:
    FREE(d_sindex);
    FREE(d_pair_mask);
    FREE(d_seg_loc);
    return err_code;
}

__host__
int GDFTdot_aow_ao_sparse(double *out, double *bra, double *ket, double *wv,
                          int ngrids, int nbas, int nbins, int npair_segs,
                          int *bas_pairs_locs, int *bas_pair2shls,
                          uint8_t *screen_index, int *ao_loc)
{
    int err_code = 0;
    int grid_blocks = (ngrids + THREADSX - 1) / THREADSX;
    int bas_blocks = (nbas + THREADSY - 1) / THREADSY;
    int tot_pairs = bas_pairs_locs[npair_segs];
    int *pair2bra = bas_pair2shls;
    int *pair2ket = bas_pair2shls + tot_pairs;
    DEVICE_INIT(uint8_t, d_sindex, screen_index, grid_blocks * bas_blocks);
    DEVICE_INIT(int, d_pair2bra, bas_pair2shls, tot_pairs * 2);
    int *d_pair2ket = d_pair2bra + tot_pairs;

    for (int seg = 0; seg < npair_segs; seg++) {
        int pair0 = bas_pairs_locs[seg];
        int pair1 = bas_pairs_locs[seg+1];
        int npairs = pair1 - pair0;
        int ish0 = pair2bra[pair0];
        int jsh0 = pair2ket[pair0];
        assert(ish0 % THREADSY == 0);
        assert(jsh0 % THREADSY == 0);
        int degen_i = ao_loc[ish0+1] - ao_loc[ish0];
        int degen_j = ao_loc[jsh0+1] - ao_loc[jsh0];
        dim3 threads(DIVXY, THREADSY, THREADSY);
        dim3 blocks((npairs+THREADSXY-1)/THREADSXY, degen_i, degen_j);
        _dot_aow_ao<<<blocks, threads>>>(out, bra, ket, wv, ngrids, nbins, d_sindex,
                                         d_pair2bra+pair0, d_pair2ket+pair0);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error of GDFTdot_aow_ao_sparse: %s\n",
                    cudaGetErrorString(err));
            err_code = 1;
            goto cleanup;
        }
    }
cleanup:
    FREE(d_sindex);
    FREE(d_pair2bra);
    return err_code;
}

__host__
int GDFTdot_ao_ao_sparse(double *out, double *bra, double *ket,
                          int ngrids, int nbas, int nbins, int npair_segs,
                          int *bas_pairs_locs, int *bas_pair2shls,
                          uint8_t *screen_index, int *ao_loc)
{
    int err_code = 0;
    int grid_blocks = (ngrids + THREADSX - 1) / THREADSX;
    int bas_blocks = (nbas + THREADSY - 1) / THREADSY;
    int tot_pairs = bas_pairs_locs[npair_segs];
    int *pair2bra = bas_pair2shls;
    int *pair2ket = bas_pair2shls + tot_pairs;
    DEVICE_INIT(uint8_t, d_sindex, screen_index, grid_blocks * bas_blocks);
    DEVICE_INIT(int, d_pair2bra, bas_pair2shls, tot_pairs * 2);
    int *d_pair2ket = d_pair2bra + tot_pairs;

    for (int seg = 0; seg < npair_segs; seg++) {
        int pair0 = bas_pairs_locs[seg];
        int pair1 = bas_pairs_locs[seg+1];
        int npairs = pair1 - pair0;
        int ish0 = pair2bra[pair0];
        int jsh0 = pair2ket[pair0];
        assert(ish0 % THREADSY == 0);
        assert(jsh0 % THREADSY == 0);
        int degen_i = ao_loc[ish0+1] - ao_loc[ish0];
        int degen_j = ao_loc[jsh0+1] - ao_loc[jsh0];
        dim3 threads(DIVXY, THREADSY, THREADSY);
        dim3 blocks((npairs+THREADSXY-1)/THREADSXY, degen_i, degen_j);
        _dot_ao_ao<<<blocks, threads>>>(out, bra, ket, ngrids, nbins, d_sindex,
                                        d_pair2bra+pair0, d_pair2ket+pair0);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error of GDFTdot_ao_ao_sparse: %s\n",
                    cudaGetErrorString(err));
            err_code = 1;
            goto cleanup;
        }
    }
cleanup:
    FREE(d_sindex);
    FREE(d_pair2bra);
    return err_code;
}
}

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "gvhf-rys/vhf.cuh"
#include "ft_ao.h"

// TODO: test kernel-15 and kernel-19 for different cases
#define GOUT_WIDTH      19
// pi^1.5
#define OVERLAP_FAC     5.56832799683170787
#define OF_COMPLEX      2

__global__
void ft_pair_kernel(double *out, AFTIntEnvVars envs, AFTBoundsInfo bounds)
{
    // sp is short for shl_pair
    int sp_block_id = blockIdx.x;
    int Gv_block_id = blockIdx.y;
    int nGv_per_block = blockDim.x;
    int gout_stride = blockDim.y;
    int nsp_per_block = blockDim.z;
    int Gv_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int sp_id = threadIdx.z;
    int npairs_ij = bounds.npairs_ij;

    if (sp_block_id * nsp_per_block + sp_id >= npairs_ij) {
        return;
    }

    int nbas = envs.nbas;
    int ish = bounds.ish_in_pair[sp_id];
    int jsh = bounds.jsh_in_pair[sp_id];
    int *sp_img_offsets = envs.shl_pair_img_offsets;
    int bas_ij = ish * nbas + jsh;
    int img0 = sp_img_offsets[bas_ij];
    int img1 = sp_img_offsets[bas_ij+1];
    if (img0 >= img1) {
        return;
    }

    int li = bounds.li;
    int lj = bounds.lj;
    int nfij = bounds.nfij;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int lij = li + lj;
    int stride_j = bounds.stride_j;
    int g_size = stride_j * (lj + 1);
    int *idx_ij = c_g_pair_idx + c_g_pair_offsets[li*LMAX1+lj];
    int *idy_ij = idx_ij + nfij;
    int *idz_ij = idy_ij + nfij;
    int *bas = envs.bas;
    double *env = envs.env;
    double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
    double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
    double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
    double *img_coords = envs.img_coords;
    int16_t *img_idx = envs.img_idx;

    int nGv = bounds.ngrids;
    double *Gv = bounds.grids + Gv_block_id * nGv_per_block;
    double kx = Gv[Gv_id];
    double ky = Gv[Gv_id + nGv];
    double kz = Gv[Gv_id + nGv * 2];
    double kk = kx * kx + ky * ky + kz * kz;

    extern __shared__ double g[];
    double *gxR = g;
    double *gxI = gxR + g_size * nGv_per_block;
    double *gyR = gxI + g_size * nGv_per_block;
    double *gyI = gyR + g_size * nGv_per_block;
    double *gzR = gyI + g_size * nGv_per_block;
    double *gzI = gzR + g_size * nGv_per_block;
    double rjri[3];
    double goutR[GOUT_WIDTH];
    double goutI[GOUT_WIDTH];
#pragma unroll
    for (int n = 0; n < GOUT_WIDTH; ++n) {
        goutR[n] = 0.;
        goutI[n] = 0.;
    }

    for (int img = img0; img < img1; img++) {
        int img_id = img_idx[img];
        double Lx = img_coords[img_id*3+0];
        double Ly = img_coords[img_id*3+1];
        double Lz = img_coords[img_id*3+2];
        double xjxi = rj[0] + Lx - ri[0];
        double yjyi = rj[1] + Ly - ri[1];
        double zjzi = rj[2] + Lz - ri[2];
        rjri[0] = xjxi;
        rjri[1] = yjyi;
        rjri[2] = zjzi;

        for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double aj_aij = aj / aij;
            double a2 = .5 / aij;
            double s0xR, s1xR, s2xR;
            double s0xI, s1xI, s2xI;
            __syncthreads();
            if (gout_id == 0) {
                double xij = rjri[0] * aj_aij + ri[0];
                double yij = rjri[1] * aj_aij + ri[1];
                double zij = rjri[2] * aj_aij + ri[2];
                double theta_ij = ai * aj_aij;
                double rr = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
                double kR = kx * xij + ky * yij + kz * zij;
                double theta_rr = theta_ij*rr + .5*a2*kk;
                double Kab = exp(-theta_rr);
                sincos(-kR, &gzI[Gv_id], &gzR[Gv_id]);
                double fac = OVERLAP_FAC * ci[ip] * cj[jp] / (aij * sqrt(aij));
                gxR[Gv_id] = fac;
                gxI[Gv_id] = 0.;
                gyR[Gv_id] = 1.;
                gyI[Gv_id] = 0.;
                // exp(-theta_rr-kR*1j)
                gzR[Gv_id] *= Kab;
                gzI[Gv_id] *= Kab;
            }

            if (lij > 0) {
                // gx[i+1] = ia2 * gx[i-1] + (rijrx[0] - kx[n]*a2*_Complex_I) * gx[i];
                __syncthreads();
                for (int n = gout_id; n < 3; n += gout_stride) {
                    double *_gxR = g + n * g_size * nGv_per_block * OF_COMPLEX;
                    double *_gxI = _gxR + g_size * nGv_per_block;
                    double RpaR = rjri[n] * aj_aij; // Rp - Ra
                    double RpaI = -a2 * Gv[Gv_id+nGv*n];
                    s0xR = _gxR[Gv_id];
                    s0xI = _gxI[Gv_id];
                    s1xR = RpaR * s0xR - RpaI * s0xI;
                    s1xI = RpaR * s0xI + RpaI * s0xR;
                    _gxR[Gv_id + nGv_per_block] = s1xR;
                    _gxI[Gv_id + nGv_per_block] = s1xI;
                    for (int i = 1; i < lij; i++) {
                        double ia2 = i * a2;
                        s2xR = ia2 * s0xR + RpaR * s1xR - RpaI * s1xI;
                        s2xI = ia2 * s0xI + RpaR * s1xI + RpaI * s1xR;
                        _gxR[Gv_id + (i+1)*nGv_per_block] = s2xR;
                        _gxI[Gv_id + (i+1)*nGv_per_block] = s2xI;
                        s0xR = s1xR;
                        s0xI = s1xI;
                        s1xR = s2xR;
                        s1xI = s2xI;
                    }
                }
            }

            // hrr
            if (lj > 0) {
                for (int n = gout_id; n < 3*OF_COMPLEX; n += gout_stride) {
                    double *_gx = g + n * g_size * nGv_per_block;
                    // The real and imaginary parts call the same expression
                    int _ix = n / 2;
                    double xjxi = rjri[_ix];
                    for (int j = 0; j < lj; ++j) {
                        int ij = (lij-j) + j*stride_j;
                        s1xR = _gx[Gv_id + ij*nGv_per_block];
                        for (--ij; ij >= j*stride_j; --ij) {
                            s0xR = _gx[sp_id + ij*nGv_per_block];
                            _gx[Gv_id + (ij+stride_j)*nGv_per_block] = s1xR - xjxi * s0xR;
                            s1xR = s0xR;
                        }
                    }
                }
            }

            __syncthreads();
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH; ++n) {
                int ij = n*gout_stride + gout_id;
                if (ij >= nfij) continue;
                int addrx = Gv_id + idx_ij[ij] * nGv_per_block;
                int addry = Gv_id + idy_ij[ij] * nGv_per_block;
                int addrz = Gv_id + idz_ij[ij] * nGv_per_block;
                double xR = gxR[addrx];
                double xI = gxI[addrx];
                double yR = gyR[addry];
                double yI = gyI[addry];
                double zR = gzR[addrz];
                double zI = gzI[addrz];
                double xyR = xR * yR - xI * yI;
                double xyI = xR * yI + xI * yR;
                goutR[n] += xyR * zR - xyI * zI;
                goutI[n] += xyR * zI + xyI * zR;
            }
        }
    }

    int nGv_in_batch = bounds.ngrids_in_batch;
    if (Gv_block_id * nGv_per_block + Gv_id < nGv_in_batch) {
        int nfi = (li + 1) * (li + 2) / 2;
        int *ao_loc = envs.ao_loc;
        int nao = ao_loc[nbas];
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        double *aft_tensor = out + ((i0*nao+j0)*nGv_in_batch
                                    + Gv_block_id*nGv_per_block) * OF_COMPLEX;
        for (int n = 0; n < GOUT_WIDTH; ++n) {
            int ij = n*gout_stride + gout_id;
            if (ij >= nfij) continue;
            int i = ij % nfi;
            int j = ij / nfi;
            int addr = (i*nao+j)*nGv_in_batch + Gv_id;
            aft_tensor[addr*2  ] = goutR[n];
            aft_tensor[addr*2+1] = goutI[n];
        }
    }
}

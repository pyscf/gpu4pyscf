/* Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <sycl/sycl.hpp>
#include "gint/gint.h"
#include "gint/sycl_alloc.hpp"
#include "nr_eval_gto.hpp"
#include "contract_rho.hpp"

#define NG_PER_BLOCK      128
#define NG_PER_THREADS    1

__attribute__((always_inline))
static void vv10_kernel(double *Fvec, double *Uvec, double *Wvec,
    const double *vvcoords, const double *coords,
    const double *W0p, const double *W0, const double *K,
    const double *Kp, const double *RpW,
    int vvngrids, int ngrids, sycl::nd_item<1>& item)
{
    // grid id
    int grid_id = static_cast<int>(item.get_global_id(0));
    const bool active = grid_id < ngrids;
    double xi, yi, zi;
    double W0i, Ki;
    if (active){
        xi = coords[grid_id];
        yi = coords[ngrids + grid_id];
        zi = coords[2*ngrids + grid_id];
        W0i = W0[grid_id];
        Ki = K[grid_id];
    }

    double F = 0.0;
    double U = 0.0;
    double W = 0.0;

    const double *xj = vvcoords;
    const double *yj = vvcoords + vvngrids;
    const double *zj = vvcoords + 2*vvngrids;

    //__shared__ double xj_smem[NG_PER_BLOCK];
    //__shared__ double yj_smem[NG_PER_BLOCK];
    //__shared__ double zj_smem[NG_PER_BLOCK];
    //__shared__ double Kp_smem[NG_PER_BLOCK];
    //__shared__ double W0p_smem[NG_PER_BLOCK];
    //__shared__ double RpW_smem[NG_PER_BLOCK];

    sycl::group thread_block = item.get_group();
    using tile_t             = sycl::double3[NG_PER_BLOCK];
    tile_t& xj_t = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);
    tile_t& kp_t = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);

    const int tx = item.get_local_id(0);

    for (int j = 0; j < vvngrids; j+=item.get_group_range(0)) {
        int idx = j + tx;
        if (idx < vvngrids){
            //xj_smem[tx] = xj[idx];
            //yj_smem[tx] = yj[idx];
            //zj_smem[tx] = zj[idx];
            //Kp_smem[tx] = Kp[idx];
            //W0p_smem[tx] = W0p[idx];
            //RpW_smem[tx] = RpW[idx];

            xj_t[tx] = {xj[idx], yj[idx], zj[idx]};
            kp_t[tx] = {Kp[idx], W0p[idx], RpW[idx]};
        }
        item.barrier(sycl::access::fence_space::local_space);

        for (int l = 0, M = std::min(NG_PER_BLOCK, vvngrids - j); l < M; ++l){
            // about 24 operations for each pair
            //double DX = xj_smem[l] - xi;//xj_tmp.x - xi;
            //double DY = yj_smem[l] - yi;//xj_tmp.y - yi;
            //double DZ = zj_smem[l] - zi;//xj_tmp.z - zi;

            sycl::double3 xj_tmp = xj_t[l];
            double DX = xj_tmp[0] - xi;
            double DY = xj_tmp[1] - yi;
            double DZ = xj_tmp[2] - zi;
            double R2 = DX*DX + DY*DY + DZ*DZ;

            sycl::double3 kp_tmp = kp_t[l]; // (Kpj, W0pj, RpWj)
            double gp = R2*kp_tmp[1] + kp_tmp[0];
            //double gp = R2 * W0p_smem[l] + Kp_smem[l];//R2*kp_tmp.y + kp_tmp.x;
            double g  = R2*W0i + Ki;
            double gt = g + gp;
            double ggt = g*gt;
            double g_gt = g + gt;
            //double T = RpW_smem[l] / (gp*ggt*ggt);//kp_tmp.z / (gp*ggt*ggt);
            double T = kp_tmp[2] / (gp*ggt*ggt);

            F += T * ggt;
            U += T * g_gt;
            W += T * R2 * g_gt;
            /*
            double ggt = g * gt;
            double ggt2 = ggt * ggt;
            double T = kp_tmp.z/(gp*ggt2);

            F += T * ggt;
            T *= (g + gt);
            U += T;
            W += T * R2;
            */
        }
        item.barrier(sycl::access::fence_space::local_space);
    }
    if(active){
        Fvec[grid_id] = F * -1.5;
        Uvec[grid_id] = U;
        Wvec[grid_id] = W;
    }

}

__attribute__((always_inline))
static void vv10_grad_kernel(double *Fvec, const double *vvcoords, const double *coords,
    const double *W0p, const double *W0,
    const double *K, const double *Kp, const double *RpW,
    int vvngrids, int ngrids, sycl::nd_item<1>& item)
{
    // grid id
    int grid_id = static_cast<int>(item.get_global_id(0));
    const bool active = grid_id < ngrids;
    double xi, yi, zi;
    double W0i, Ki;
    if (active){
        xi = coords[grid_id];
        yi = coords[ngrids + grid_id];
        zi = coords[2*ngrids + grid_id];
        W0i = W0[grid_id];
        Ki = K[grid_id];
    }
    double FX = 0;
    double FY = 0;
    double FZ = 0;

    const double *xj = vvcoords;
    const double *yj = vvcoords + vvngrids;
    const double *zj = vvcoords + 2*vvngrids;

    sycl::group thread_block = item.get_group();
    using tile_t             = sycl::double3[NG_PER_BLOCK];
    tile_t& xj_t = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);
    tile_t& kp_t = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);

    const int tx = item.get_local_id(0);
    for (int j = 0; j < vvngrids; j+=item.get_group_range(0)) {
        int idx = j + item.get_local_id(0);
        if (idx < vvngrids){
            xj_t[tx] = {xj[idx], yj[idx], zj[idx]};
            kp_t[tx] = {Kp[idx], W0p[idx], RpW[idx]};
        }
        item.barrier(sycl::access::fence_space::local_space);
        for (int l = 0, M = std::min(NG_PER_BLOCK, vvngrids - j); l < M; ++l){
            sycl::double3 xj_tmp = xj_t[l];
            // about 23 operations for each pair
            double DX = xj_tmp[0] - xi;
            double DY = xj_tmp[1] - yi;
            double DZ = xj_tmp[2] - zi;
            double R2 = DX*DX + DY*DY + DZ*DZ;

            sycl::double3 kp_tmp = kp_t[l];
            double gp = R2*kp_tmp[1] + kp_tmp[0];
            double g  = R2*W0i + Ki;
            double gt = g + gp;
            double ggp = g * gp;
            double ggt_gp = gt * ggp;
            double T = kp_tmp[2] / (ggt_gp * ggt_gp);
            double Q = T * ((W0i*gp + kp_tmp[1]*g)*gt + (W0i+kp_tmp[1])*ggp);

            FX += Q * DX;
            FY += Q * DY;
            FZ += Q * DZ;
        }
         item.barrier(sycl::access::fence_space::local_space);
    }
    if (active) {
        Fvec[0*ngrids + grid_id] = FX * -3;
        Fvec[1*ngrids + grid_id] = FY * -3;
        Fvec[2*ngrids + grid_id] = FZ * -3;
    }
}

extern "C" {
int VXC_vv10nlc(sycl::queue& stream, double *Fvec, double *Uvec, double *Wvec,
                 const double *vvcoords, const double *coords,
                 const double *W0p, const double *W0, const double *K,
                 const double *Kp, const double *RpW,
                 int vvngrids, int ngrids)
{
    sycl::range<1> threads(NG_PER_BLOCK);
    sycl::range<1> blocks((ngrids/NG_PER_THREADS+1+NG_PER_BLOCK-1)/NG_PER_BLOCK);
    stream.parallel_for(sycl::nd_range<1>(blocks * threads, threads), [=](auto item) {
	vv10_kernel(Fvec, Uvec, Wvec,
		    vvcoords, coords,
		    W0p, W0, K, Kp, RpW, vvngrids, ngrids, item);
    });
    return 0;
}

int VXC_vv10nlc_grad(sycl::queue& stream, double *Fvec,
                    const double *vvcoords, const double *coords,
                    const double *W0p, const double *W0, const double *K,
                    const double *Kp, const double *RpW,
                    int vvngrids, int ngrids)
{
    sycl::range<1> threads(NG_PER_BLOCK);
    sycl::range<1> blocks((ngrids+NG_PER_BLOCK-1)/NG_PER_BLOCK);
    stream.parallel_for(sycl::nd_range<1>(blocks * threads, threads), [=](auto item) {
	vv10_grad_kernel(Fvec, vvcoords, coords, W0p, W0, K, Kp, RpW, vvngrids, ngrids, item);
    });
    return 0;
}
}

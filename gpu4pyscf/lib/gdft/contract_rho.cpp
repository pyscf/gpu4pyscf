/*
 * gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
 *
 * Copyright (C) 2022 Qiming Sun
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
#include <sycl/sycl.hpp>
#include "contract_rho.hpp"
// TODO: improve this?
__attribute__((always_inline))
void GDFTcontract_rho_kernel(double *rho, double *bra, double *ket, int ngrids, int nao, sycl::nd_item<2>& item)
{
    int grid_id = static_cast<int>( item.get_global_id(1) );
    const bool active = grid_id < ngrids;
    size_t Ngrids = ngrids;
    double v = 0;
    if (active){
        for (int ao_id = static_cast<int>(item.get_local_id(0)); ao_id < nao; ao_id += BLKSIZEY) {
            int ket_idx = grid_id + ao_id * Ngrids;
            v += bra[ket_idx] * ket[ket_idx];
        }
    }

    sycl::group thread_block = item.get_group();
    using tile_t             = double[BLKSIZEX*(BLKSIZEY+1)];
    tile_t& buf = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);
    int ix = static_cast<int>(item.get_local_id(1));
    int iy = static_cast<int>(item.get_local_id(0));
    int ixy = ix + BLKSIZEX * iy;
    buf[ixy] = v;   item.barrier(sycl::access::fence_space::local_space);

    if (static_cast<int>( thread_block.get_local_range(0) ) >= 32 && iy < 16) buf[ixy] += buf[ixy + BLKSIZEX * 16]; item.barrier(sycl::access::fence_space::local_space);
    if (static_cast<int>( thread_block.get_local_range(0) ) >= 16 && iy < 8)  buf[ixy] += buf[ixy + BLKSIZEX * 8];  item.barrier(sycl::access::fence_space::local_space);
    if (static_cast<int>( thread_block.get_local_range(0) ) >= 8  && iy < 4)  buf[ixy] += buf[ixy + BLKSIZEX * 4];  item.barrier(sycl::access::fence_space::local_space);
    if (static_cast<int>( thread_block.get_local_range(0) ) >= 4  && iy < 2)  buf[ixy] += buf[ixy + BLKSIZEX * 2];  item.barrier(sycl::access::fence_space::local_space);
    if (static_cast<int>( thread_block.get_local_range(0) ) >= 2  && iy < 1)  buf[ixy] += buf[ixy + BLKSIZEX * 1];  item.barrier(sycl::access::fence_space::local_space);

    if (iy == 0 && active) {
        rho[grid_id] = buf[ix];
    }
}

__attribute__((always_inline))
void GDFTcontract_rho4_kernel(double *rho, double *bra, double *ket, int ngrids, int nao, int count, sycl::nd_item<2>& item)
{
    int grid_id = static_cast<int>( item.get_global_id(1) );
    const bool active = grid_id < ngrids;
    size_t ket_stride = nao * ngrids;
    size_t rho_stride = count * ngrids;


    sycl::group thread_block = item.get_group();
    using tile_t             = double[BLKSIZEX*(BLKSIZEY+1)];
    tile_t& buf = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);    

    for (int ia = 0; ia < count; ia++){
        double v[4] = {0.0, 0.0, 0.0, 0.0};
        if (active){
            for (int ao_id = static_cast<int>(item.get_local_id(0)); ao_id < nao; ao_id += BLKSIZEY) {
                int ket_idx = grid_id + ao_id * ngrids;
                double bra_tmp = bra[ket_idx + ia * ket_stride];
                v[0] += bra_tmp * ket[0*ket_stride + ket_idx];
                v[1] += bra_tmp * ket[1*ket_stride + ket_idx];
                v[2] += bra_tmp * ket[2*ket_stride + ket_idx];
                v[3] += bra_tmp * ket[3*ket_stride + ket_idx];
            }
        }

        int ix = static_cast<int>(item.get_local_id(1));
        int iy = static_cast<int>(item.get_local_id(0));
        int ixy = ix + BLKSIZEX * iy;
        for (int i = 0; i < 4; i++){
            buf[ixy] = v[i];   item.barrier(sycl::access::fence_space::local_space);
            if (static_cast<int>( thread_block.get_local_range(0) ) >= 32 && iy < 16) buf[ixy] += buf[ixy + BLKSIZEX * 16]; item.barrier(sycl::access::fence_space::local_space);
            if (static_cast<int>( thread_block.get_local_range(0) ) >= 16 && iy < 8)  buf[ixy] += buf[ixy + BLKSIZEX * 8];  item.barrier(sycl::access::fence_space::local_space);
            if (static_cast<int>( thread_block.get_local_range(0) ) >= 8  && iy < 4)  buf[ixy] += buf[ixy + BLKSIZEX * 4];  item.barrier(sycl::access::fence_space::local_space);
            if (static_cast<int>( thread_block.get_local_range(0) ) >= 4  && iy < 2)  buf[ixy] += buf[ixy + BLKSIZEX * 2];  item.barrier(sycl::access::fence_space::local_space);
            if (static_cast<int>( thread_block.get_local_range(0) ) >= 2  && iy < 1)  buf[ixy] += buf[ixy + BLKSIZEX * 1];  item.barrier(sycl::access::fence_space::local_space);

            if (iy == 0 && active) {
                rho[grid_id + ia * ngrids + rho_stride * i] = buf[ix];
            }
        }
    }
}

__attribute__((always_inline))
void GDFTcontract_rho_gga_kernel(double *rho, double *bra, double *ket, int ngrids, int nao, sycl::nd_item<2>& item)
{
    int grid_id = static_cast<int>( item.get_global_id(1) );
    const bool active = grid_id < ngrids;

    size_t Ngrids = ngrids;
    size_t ket_stride = nao * ngrids;

    double v[4] = {0.0, 0.0, 0.0, 0.0};
    if (active){
        for (int ao_id = static_cast<int>(item.get_local_id(0)); ao_id < nao; ao_id += BLKSIZEY) {
            int ket_idx = grid_id + ao_id * Ngrids;
            double bra_tmp = bra[ket_idx];
            double ket_tmp = ket[ket_idx];

            v[0] += bra_tmp * ket_tmp;

            ket_idx += ket_stride;
            v[1] += bra_tmp * ket[ket_idx];
            v[1] += ket_tmp * bra[ket_idx];

            ket_idx += ket_stride;
            v[2] += bra_tmp * ket[ket_idx];
            v[2] += ket_tmp * bra[ket_idx];

            ket_idx += ket_stride;
            v[3] += bra_tmp * ket[ket_idx];
            v[3] += ket_tmp * bra[ket_idx];
        }
    }

    sycl::group thread_block = item.get_group();
    using tile_t             = double[BLKSIZEX*(BLKSIZEY+1)];
    tile_t& buf = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);
    int ix = static_cast<int>(item.get_local_id(1));
    int iy = static_cast<int>(item.get_local_id(0));
    int ixy = ix + BLKSIZEX * iy;

    for (int i = 0; i < 4; i++){
        buf[ixy] = v[i];   item.barrier(sycl::access::fence_space::local_space);
        if (static_cast<int>( thread_block.get_local_range(0) ) >= 32 && iy < 16) buf[ixy] += buf[ixy + BLKSIZEX * 16]; item.barrier(sycl::access::fence_space::local_space);
        if (static_cast<int>( thread_block.get_local_range(0) ) >= 16 && iy < 8)  buf[ixy] += buf[ixy + BLKSIZEX * 8];  item.barrier(sycl::access::fence_space::local_space);
        if (static_cast<int>( thread_block.get_local_range(0) ) >= 8  && iy < 4)  buf[ixy] += buf[ixy + BLKSIZEX * 4];  item.barrier(sycl::access::fence_space::local_space);
        if (static_cast<int>( thread_block.get_local_range(0) ) >= 4  && iy < 2)  buf[ixy] += buf[ixy + BLKSIZEX * 2];  item.barrier(sycl::access::fence_space::local_space);
        if (static_cast<int>( thread_block.get_local_range(0) ) >= 2  && iy < 1)  buf[ixy] += buf[ixy + BLKSIZEX * 1];  item.barrier(sycl::access::fence_space::local_space);

        if (iy == 0 && active) {
            rho[grid_id + ngrids * i] = 2.0 * buf[ix];
        }
    }
}


__attribute__((always_inline))
void GDFTcontract_rho_mgga_kernel(double *rho, double *bra, double *ket, int ngrids, int nao, sycl::nd_item<2>& item)
{
    int grid_id = static_cast<int>( item.get_global_id(1) );
    const bool active = grid_id < ngrids;

    size_t Ngrids = ngrids;
    size_t ket_stride = nao * ngrids;

    double v[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    if (active){
        for (int ao_id = static_cast<int>(item.get_local_id(0)); ao_id < nao; ao_id += BLKSIZEY) {
            int ket_idx = grid_id + ao_id * Ngrids;
            double bra_tmp0 = bra[ket_idx];
            double ket_tmp0 = ket[ket_idx];

            v[0] += bra_tmp0 * ket_tmp0;

            ket_idx += ket_stride;
            double bra_tmp1 = bra[ket_idx];
            double ket_tmp1 = ket[ket_idx];
            v[1] += bra_tmp0 * ket_tmp1;
            v[1] += ket_tmp0 * bra_tmp1;
            v[4] += bra_tmp1 * ket_tmp1;

            ket_idx += ket_stride;
            bra_tmp1 = bra[ket_idx];
            ket_tmp1 = ket[ket_idx];
            v[2] += bra_tmp0 * ket_tmp1;
            v[2] += ket_tmp0 * bra_tmp1;
            v[4] += bra_tmp1 * ket_tmp1;

            ket_idx += ket_stride;
            bra_tmp1 = bra[ket_idx];
            ket_tmp1 = ket[ket_idx];
            v[3] += bra_tmp0 * ket_tmp1;
            v[3] += ket_tmp0 * bra_tmp1;
            v[4] += bra_tmp1 * ket_tmp1;

        }
    }

    v[4] *= 0.5;

    sycl::group thread_block = item.get_group();
    using tile_t             = double[BLKSIZEX*(BLKSIZEY+1)];
    tile_t& buf = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);
    int ix = static_cast<int>(item.get_local_id(1));
    int iy = static_cast<int>(item.get_local_id(0));
    int ixy = ix + BLKSIZEX * iy;

    for (int i = 0; i < 5; i++){
        buf[ixy] = v[i];   item.barrier(sycl::access::fence_space::local_space);
        if (static_cast<int>( thread_block.get_local_range(0) ) >= 32 && iy < 16) buf[ixy] += buf[ixy + BLKSIZEX * 16]; item.barrier(sycl::access::fence_space::local_space);
        if (static_cast<int>( thread_block.get_local_range(0) ) >= 16 && iy < 8)  buf[ixy] += buf[ixy + BLKSIZEX * 8];  item.barrier(sycl::access::fence_space::local_space);
        if (static_cast<int>( thread_block.get_local_range(0) ) >= 8  && iy < 4)  buf[ixy] += buf[ixy + BLKSIZEX * 4];  item.barrier(sycl::access::fence_space::local_space);
        if (static_cast<int>( thread_block.get_local_range(0) ) >= 4  && iy < 2)  buf[ixy] += buf[ixy + BLKSIZEX * 2];  item.barrier(sycl::access::fence_space::local_space);
        if (static_cast<int>( thread_block.get_local_range(0) ) >= 2  && iy < 1)  buf[ixy] += buf[ixy + BLKSIZEX * 1];  item.barrier(sycl::access::fence_space::local_space);

        if (iy == 0 && active) {
            rho[grid_id + ngrids * i] = 2.0 * buf[ix];
        }
    }
}

__attribute__((always_inline))
void GDFTscale_ao_kernel(double *out, double *ket, double *wv,
                         int ngrids, int nao, int nvar, sycl::nd_item<2>& item)
{
    int grid_id = static_cast<int>( item.get_global_id(1) );
    int ao_id = static_cast<int>( item.get_global_id(0) );
    if (grid_id >= ngrids || ao_id >= nao) {
        return;
    }

    size_t Ngrids = ngrids;
    size_t Nag = nao * Ngrids;
    size_t ixy = grid_id + ao_id * Ngrids;
    double val = 0;
    int n;
    for (n = 0; n < nvar; ++n) {
         val += ket[ixy + Nag * n] * wv[grid_id + ngrids * n];
    }
    out[ixy] = val;
}

__attribute__((always_inline))
void GDFT_make_dR_dao_w_kernel(double *out, double *ket, double *wv,
			       int ngrids, int nao, sycl::nd_item<2>& item)
{
    int grid_id = static_cast<int>( item.get_global_id(1) );
    int ao_id = static_cast<int>( item.get_global_id(0) );
    if (grid_id >= ngrids || ao_id >= nao) {
        return;
    }

    size_t Ngrids = ngrids;
    size_t Nag = nao * Ngrids;
    size_t ixy = grid_id + ao_id * Ngrids;

    double wv0 = wv[grid_id + ngrids * 0];
    double wv1 = wv[grid_id + ngrids * 1];
    double wv2 = wv[grid_id + ngrids * 2];
    double wv3 = wv[grid_id + ngrids * 3];

    double ket5 = ket[ixy + Nag * 5];
    double ket6 = ket[ixy + Nag * 6];
    double val;
    val = ket[ixy + Nag * 1] * wv0;
    val+= ket[ixy + Nag * 4] * wv1;
    val+= ket5 * wv2;
    val+= ket6 * wv3;
    out[ixy + Nag * 0] = val;

    double ket8 = ket[ixy + Nag * 8];
    val = ket[ixy + Nag * 2] * wv0;
    val+= ket5 * wv1;
    val+= ket[ixy + Nag * 7] * wv2;
    val+= ket8 * wv3;
    out[ixy + Nag * 1] = val;

    val = ket[ixy + Nag * 3] * wv0;
    val+= ket6 * wv1;
    val+= ket8 * wv2;
    val+= ket[ixy + Nag * 9] * wv3;
    out[ixy + Nag * 2] = val;
}


extern "C"{

int GDFTcontract_rho(sycl::queue& stream, double *rho, double *bra, double *ket, int ngrids, int nao)
{
    sycl::range<2> threads(BLKSIZEY, BLKSIZEX);
    sycl::range<2> blocks(1, (ngrids+BLKSIZEX-1)/BLKSIZEX);
    stream.parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GDFTcontract_rho_kernel(rho, bra, ket, ngrids, nao, item); });
    return 0;
}

int GDFTcontract_rho4(sycl::queue& stream, double *rho, double *bra, double *ket, int ngrids, int nao, int count)
{
    sycl::range<2> threads(BLKSIZEY, BLKSIZEX);
    sycl::range<2> blocks(1, (ngrids+BLKSIZEX-1)/BLKSIZEX);
    stream.parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GDFTcontract_rho4_kernel(rho, bra, ket, ngrids, nao, count, item); });
    return 0;
}

int GDFTcontract_rho_gga(sycl::queue& stream, double *rho, double *bra, double *ket, int ngrids, int nao)
{
    sycl::range<2> threads(BLKSIZEY, BLKSIZEX);
    sycl::range<2> blocks(1, (ngrids+BLKSIZEX-1)/BLKSIZEX);
    stream.parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GDFTcontract_rho_gga_kernel(rho, bra, ket, ngrids, nao, item); });
    return 0;
}

int GDFTcontract_rho_mgga(sycl::queue& stream, double *rho, double *bra, double *ket, int ngrids, int nao)
{
    sycl::range<2> threads(BLKSIZEY, BLKSIZEX);
    sycl::range<2> blocks(1, (ngrids+BLKSIZEX-1)/BLKSIZEX);
    stream.parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GDFTcontract_rho_mgga_kernel(rho, bra, ket, ngrids, nao, item); });
    return 0;
}

int GDFT_make_dR_dao_w(sycl::queue& stream, double *out, double *ket, double *wv,
                 int ngrids, int nao)
{
    sycl::range<2> threads(BLKSIZEY, BLKSIZEX);
    sycl::range<2> blocks((nao+BLKSIZEY-1)/BLKSIZEY, (ngrids+BLKSIZEX-1)/BLKSIZEX);
    stream.parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GDFT_make_dR_dao_w_kernel(out, ket, wv, ngrids, nao, item); });
    return 0;
}

int GDFTscale_ao(sycl::queue& stream, double *out, double *ket, double *wv,
                 int ngrids, int nao, int nvar)
{
    sycl::range<2> threads(BLKSIZEY, BLKSIZEX);
    sycl::range<2> blocks((nao+BLKSIZEY-1)/BLKSIZEY, (ngrids+BLKSIZEX-1)/BLKSIZEX);
    stream.parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GDFTscale_ao_kernel(out, ket, wv, ngrids, nao, nvar, item); });
    return 0;
}

}

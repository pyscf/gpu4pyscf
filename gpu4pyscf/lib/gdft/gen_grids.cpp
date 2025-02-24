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
#include <stdlib.h>
#include <math.h>
#include <sycl/sycl.hpp>

#define NATOM_PER_BLOCK        128

__attribute__((always_inline))
void GDFTgen_grid_kernel(double *pbecke, const double *coords, const double *atm_coords, const double *a,
			 int ngrids, int natm, sycl::nd_item<1>& item)
{
    int grid_id = static_cast<int>( item.get_global_id(0) );
    const bool active = grid_id < ngrids;
    double xg = 0.0;
    double yg = 0.0;
    double zg = 0.0;
    if(active){
        xg = coords[3*grid_id+0];
        yg = coords[3*grid_id+1];
        zg = coords[3*grid_id+2];
    }
    sycl::group thread_block = item.get_group();
    using tile_t             = double[NATOM_PER_BLOCK];
    tile_t& xj = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);
    tile_t& yj = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);
    tile_t& zj = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);
    tile_t& a_smem = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);
    tile_t& dij_smem = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);
    const int tx = static_cast<int>( item.get_local_id(0) );

    for (int atom_i = 0; atom_i < natm; atom_i++){
        double xi = atm_coords[atom_i];
        double yi = atm_coords[atom_i + natm];
        double zi = atm_coords[atom_i + 2*natm];

        double becke = 2.0;
        double dx, dy, dz, dig;
        if (active){
            // distance between grids and atom i
            dx = xg - xi;
            dy = yg - yi;
            dz = zg - zi;
            dig = std::sqrt(dx * dx + dy * dy + dz * dz);
        }
        for (int j = 0; j < natm; j+=static_cast<int>( thread_block.get_local_range(0) )){
            int atom_idx = j + tx;
            if (atom_idx < natm){
                double xj_t = atm_coords[atom_idx];
                double yj_t = atm_coords[atom_idx + natm];
                double zj_t = atm_coords[atom_idx + 2*natm];

                // distance between atom i and atom j
                dx = xi - xj_t;
                dy = yi - yj_t;
                dz = zi - zj_t;
                // double dij = rnorm3d(dx, dy, dz);
                double dij = std::sqrt(dx * dx + dy * dy + dz * dz);

                // distance between atom i and atom j
                dij_smem[tx] = dij;
                xj[tx] = xj_t;
                yj[tx] = yj_t;
                zj[tx] = zj_t;
                a_smem[tx] = a[atom_i * natm + atom_idx];
            }
            item.barrier(sycl::access::fence_space::local_space);

            for (int l = 0, M = std::min(NATOM_PER_BLOCK, natm-j); l < M; ++l){
                int atom_j = j + l;
                // distance between grids and atom j
                dx = xg - xj[l];
                dy = yg - yj[l];
                dz = zg - zj[l];
                // double djg = norm3d(dx, dy, dz);
                double djg = std::sqrt(dx * dx + dy * dy + dz * dz);

                double dij = dij_smem[l];
                double aij = a_smem[l];
                double g = (atom_i == atom_j) ? 0.0 : (dig - djg) * dij;

                // atomic radii adjust function
                double g1 = g*g - 1.0;
                //g1 -= 1.0;
                g += g1 * aij;

                // becke scheme
                g = (3.0 - g*g) * g * .5;
                g = (3.0 - g*g) * g * .5;
                g = (3.0 - g*g) * g * .5;

                g = 0.5 * (1.0 - g);
                becke *= g;
            }
            item.barrier(sycl::access::fence_space::local_space);
        }
        if(active){
            pbecke[atom_i*ngrids + grid_id] = becke;
        }
    }
}

__attribute__((always_inline))
void GDFTgroup_grids_kernel(int* group_ids, const double* atom_coords, const double* coords, int natm, int ngrids, sycl::nd_item<1>& item){
    int grid_id = static_cast<int>( item.get_global_id(0) );

    double xg = coords[grid_id];
    double yg = coords[grid_id + ngrids];
    double zg = coords[grid_id + 2*ngrids];

    double r2min = 1e30;
    int idx = 0;
    const int tx = static_cast<int>( item.get_local_id(0) );
    sycl::group thread_block = item.get_group();
    using tile_t             = double[NATOM_PER_BLOCK];
    tile_t& x_atom = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);
    tile_t& y_atom = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);
    tile_t& z_atom = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);
    for (int j = 0; j < natm; j+=static_cast<int>( thread_block.get_local_range(0) )){
        int atom_idx = j + tx;
        if (atom_idx < natm){
            // distance between atom i and atom j
            x_atom[tx] = atom_coords[atom_idx];
            y_atom[tx] = atom_coords[atom_idx + natm];
            z_atom[tx] = atom_coords[atom_idx + 2*natm];
        }
        item.barrier(sycl::access::fence_space::local_space);

        for (int l = 0, M = std::min(NATOM_PER_BLOCK, natm-j); l < M; ++l){
            int atom_j = j + l;
            double xa = x_atom[l] - xg;
            double ya = y_atom[l] - yg;
            double za = z_atom[l] - zg;
            double r2 = xa*xa + ya*ya + za*za;
            if (r2 < r2min){
                r2min = r2;
                idx = atom_j;
            }
        }
    }
    group_ids[grid_id] = idx;
}

extern "C"{
int GDFTgen_grid_partition(sycl::queue& stream, double *pbecke,
const double *coords, const double *atm_coords, const double *a, int ngrids, int natm)
{
    sycl::range<1> threads(NATOM_PER_BLOCK);
    sycl::range<1> blocks((ngrids+NATOM_PER_BLOCK-1)/NATOM_PER_BLOCK);
    stream.parallel_for(sycl::nd_range<1>(blocks * threads, threads), [=](auto item) {
	GDFTgen_grid_kernel(pbecke, coords, atm_coords, a, ngrids, natm, item);
    });
    return 0;
}

int GDFTgroup_grids(sycl::queue& stream, int* group_ids, const double* atom_coords, const double* coords,
    int natm, int ngrids){
    if (ngrids % NATOM_PER_BLOCK != 0){
        fprintf(stderr, "SYCL Error of gen grids: grids alignment must be %d.", NATOM_PER_BLOCK);
        return 1;
    }
    sycl::range<1> threads(NATOM_PER_BLOCK);
    sycl::range<1> blocks((ngrids+NATOM_PER_BLOCK-1)/NATOM_PER_BLOCK);
    stream.parallel_for(sycl::nd_range<1>(blocks * threads, threads), [=](auto item) {
	GDFTgroup_grids_kernel(group_ids, atom_coords, coords, natm, ngrids, item);
    });
    return 0;
}

}

/*
 * Copyright 2025 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// =====================================================================
// Per-kernel-family constants for the md_j contraction.
//
// MD_J_TILE_THREADS
//   Tile-row width used by the unrolled md_j_*_* kernels. The launch
//   geometry for those kernels is `dim3 threads(16, 16)` and the
//   per-row task batching uses `sq_id = tx + 16*ty`. The qd_offset
//   layered pyramid is therefore indexed at this same width when the
//   tile16 unrolled path is selected.
//
// MD_J_QD_ALIGN
//   Alignment unit (in elements) for the per-power-of-two strata of
//   the qd_*_max pyramid in qd_offset_for_threads(). Set to the warp
//   width assumed by the kernels that consume qd_*_max so the strided
//   accesses are warp-/cache-line-friendly. `warpSize` is a
//   device-only built-in; this host-side constant mirrors it.
//   Centralizing the value here makes a future wider-wavefront port
//   a single-point change.
// =====================================================================
#ifndef MD_J_TILE_THREADS
#define MD_J_TILE_THREADS 16
#endif

#ifndef MD_J_QD_ALIGN
#define MD_J_QD_ALIGN     32
#endif

typedef struct {
    int li;
    int lj;
    int lk;
    int ll;
    int lij;
    int lkl;
    int order;
    int nf3ij;
    int nf3kl;
    int nf3ijkl;
    int npairs_ij;
    int npairs_kl;
    int *pair_ij_mapping; // the significant ij pairs, mapping to i*nao+j
    int *pair_kl_mapping;
    int *pair_ij_loc; // offsets to the input dm_xyz for each ij pair
    int *pair_kl_loc;
    float *qd_ij_max; // largest dm_cond*q_cond within each block for ij pair
    float *qd_kl_max;
    float *q_cond;
    float cutoff; // cutoff to screening schwarz estimation q_ij+q_kl
} MDBoundsInfo;

int offset_for_Rt2_idx(int lij, int lkl);
int qd_offset_for_threads(int npairs, int threads);

extern __device__ int Rt2_idx_offsets[];
extern __device__ uint16_t Rt2_ij_kl[];
extern __device__ uint16_t Rt2_kl_ij[];
extern __constant__ int8_t c_Rt2_efg_phase[];
extern __constant__ int8_t c_Rt_tuv_fac[];
extern __constant__ uint16_t c_Rt_idx[];

/*
 * Copyright 2024-2025 The PySCF Developers. All Rights Reserved.
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

#include <stdint.h>

// WARP_SIZE: compile-time constant used for shared-memory sizing.
// `warpSize` (HIP/CUDA device-runtime built-in) is not constexpr,
// so we keep a literal here. Guarded so the build can override
// it (e.g. -DWARP_SIZE=64) for future wider-wavefront targets.
#ifndef WARP_SIZE
#define WARP_SIZE       32
#endif
#define WARPS           8
// Fixed at 128 (not WARP_SIZE*4): ft_ao_bdiv_kernel and the ip1 kernels tile the
// Gv axis by the fixed NG_PER_BLOCK=32 below, so the block size must not scale
// with the wavefront -- otherwise host launch (host pass WARP_SIZE=32) and device
// (per-arch WARP_SIZE) disagree, and a wave64 block of 256 doubles the static
// shared arrays past the 64 KB LDS limit. CUDA is unchanged (WARP_SIZE==32 -> 128).
#define FT_AO_THREADS   128
#define NG_PER_BLOCK    32
#define AUXL            6
#define AUXNF           ((AUXL+1)*(AUXL+2)/2)
// pi^1.5
#define OVERLAP_FAC     5.56832799683170787
#define OF_COMPLEX      2

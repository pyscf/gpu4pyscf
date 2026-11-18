/*
 * Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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

/*
 * Per-arch wavefront width for a single binary that runs on any AMD GPU.
 *
 * This header is force-included (-include) into every HIP translation unit, so it
 * is the single source of truth for WARP_SIZE. On CUDA/NVIDIA it is pulled in
 * explicitly (relative include) by the few headers that need the host helper.
 *
 * WARP_SIZE -- the wavefront width the *device* code is compiled for. In a
 * multi-arch (fat-binary) HIP build each device sub-compilation defines the
 * target __gfxNNN__ macro, so WARP_SIZE resolves per-arch: 64 on GCN/CDNA
 * (gfx9xx, e.g. MI2xx/MI3xx), 32 on RDNA and newer (gfx10xx/11xx/12xx). The host
 * sub-compilation (and CUDA/NVIDIA) define no __gfxNNN__ macro and fall back to
 * 32. A command-line -DWARP_SIZE=... (USE_WAVE64) overrides everything.
 *
 * gpu4pyscf_effective_warp_size() -- the width to use for *host-side* kernel
 * launch / shared-memory / scratch sizing. Host code cannot see the per-arch
 * __gfxNNN__ macro, so it asks the running device and returns
 * max(compile-time WARP_SIZE, runtime warpSize): the native width on an auto
 * build, and the forced width under USE_WAVE64 (which sets -DWARP_SIZE=64 while
 * the device still reports its native warpSize). A single active device is
 * assumed (the result is cached).
 */
#ifndef GPU4PYSCF_WARP_SIZE_H
#define GPU4PYSCF_WARP_SIZE_H

#ifndef WARP_SIZE
#  if defined(__gfx900__) || defined(__gfx902__) || defined(__gfx904__) || \
      defined(__gfx906__) || defined(__gfx908__) || defined(__gfx909__) || \
      defined(__gfx90a__) || defined(__gfx90c__) || defined(__gfx940__) || \
      defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)
#    define WARP_SIZE 64
#  else
#    define WARP_SIZE 32
#  endif
#endif

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
static inline __attribute__((unused)) int gpu4pyscf_effective_warp_size(void)
{
    static int cached = 0;
    if (cached != 0) {
        return cached;
    }
    int rt = 32;
    int dev = 0;
    if (hipGetDevice(&dev) == hipSuccess) {
        hipDeviceProp_t prop;
        if (hipGetDeviceProperties(&prop, dev) == hipSuccess && prop.warpSize > 0) {
            rt = prop.warpSize;
        }
    }
    cached = (rt > WARP_SIZE) ? rt : WARP_SIZE;
    return cached;
}
#else
static inline __attribute__((unused)) int gpu4pyscf_effective_warp_size(void)
{
    return WARP_SIZE;  /* CUDA/NVIDIA: 32, compile-time */
}
#endif

#endif  /* GPU4PYSCF_WARP_SIZE_H */

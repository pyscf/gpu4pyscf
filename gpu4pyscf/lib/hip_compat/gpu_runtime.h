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
 * CUDA -> HIP compatibility shim.
 *
 * gpu4pyscf's device kernels are authored against the CUDA runtime. To build
 * them unmodified with hipcc/ROCm, this header is force-included for HIP
 * compiles (see -include in the top-level CMakeLists) and is also pulled in by
 * the cuda_runtime.h / cuda.h stubs in this directory, so existing
 * `#include <cuda_runtime.h>` / `#include <cuda.h>` lines resolve.
 *
 * Only the CUDA runtime surface actually used by gpu4pyscf is mapped. The
 * whole file is a no-op on NVIDIA/CUDA builds, so the CUDA path is unchanged.
 *
 * When adding a newly-used CUDA runtime symbol, map it here rather than editing
 * the kernel sources, keeping a single portable source tree.
 */

#ifndef GPU4PYSCF_HIP_COMPAT_GPU_RUNTIME_H
#define GPU4PYSCF_HIP_COMPAT_GPU_RUNTIME_H

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__) || defined(USE_HIP)

#include <hip/hip_runtime.h>
// Pre-include the HIP headers that provide their own overloads of the warp
// *_sync primitives (fp16/bf16). Processing them here -- before the
// mask-widening macros below are defined -- means their definitions stay intact
// (subsequent includes are guarded), so the macros only ever rewrite
// gpu4pyscf's own call sites, not HIP/hipCUB library code.
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

// Some sources report or version-gate on CUDA_VERSION. Define it to 0 on HIP so
// CUDA-version-gated code paths (e.g. `#if CUDA_VERSION >= 12040` guarding the
// CUDA-only __maxnreg__ launch-bound attribute) are disabled.
#ifndef CUDA_VERSION
#define CUDA_VERSION 0
#endif

// HIP's warp *_sync primitives take a 64-bit lane mask (wavefronts may be 64
// wide) and static_assert that the mask type is 64-bit to catch accidental
// implicit promotion. The CUDA sources pass 32-bit literals such as 0xffffffff.
// Widen the mask argument; the value is preserved (upper bits are unused on
// wave32). The self-referential macro name is not re-expanded, so these resolve
// to the real HIP builtins.
#define __shfl_sync(mask, ...)       __shfl_sync((unsigned long long)(mask), __VA_ARGS__)
#define __shfl_up_sync(mask, ...)    __shfl_up_sync((unsigned long long)(mask), __VA_ARGS__)
#define __shfl_down_sync(mask, ...)  __shfl_down_sync((unsigned long long)(mask), __VA_ARGS__)
#define __shfl_xor_sync(mask, ...)   __shfl_xor_sync((unsigned long long)(mask), __VA_ARGS__)
#define __ballot_sync(mask, ...)     __ballot_sync((unsigned long long)(mask), __VA_ARGS__)
#define __any_sync(mask, ...)        __any_sync((unsigned long long)(mask), __VA_ARGS__)
#define __all_sync(mask, ...)        __all_sync((unsigned long long)(mask), __VA_ARGS__)

/* --- error handling --- */
#define cudaError_t                                 hipError_t
#define cudaError                                   hipError_t
#define cudaSuccess                                 hipSuccess
#define cudaGetLastError                            hipGetLastError
#define cudaPeekAtLastError                         hipPeekAtLastError
#define cudaGetErrorString                          hipGetErrorString
#define cudaGetErrorName                            hipGetErrorName

/* --- streams / events / sync --- */
#define cudaStream_t                                hipStream_t
#define cudaEvent_t                                 hipEvent_t
#define cudaStreamSynchronize                       hipStreamSynchronize
#define cudaStreamCreate                            hipStreamCreate
#define cudaStreamDestroy                           hipStreamDestroy
#define cudaDeviceSynchronize                       hipDeviceSynchronize
#define cudaDeviceReset                             hipDeviceReset

/* --- memory --- */
#define cudaMalloc                                  hipMalloc
#define cudaFree                                    hipFree
#define cudaMemset                                  hipMemset
#define cudaMemsetAsync                             hipMemsetAsync
#define cudaMemcpy                                  hipMemcpy
#define cudaMemcpyAsync                             hipMemcpyAsync
#define cudaMemcpy2D                                hipMemcpy2D
#define cudaMemcpy2DAsync                           hipMemcpy2DAsync
#define cudaMemcpyToSymbol                          hipMemcpyToSymbol
#define cudaMemcpyFromSymbol                        hipMemcpyFromSymbol
#define cudaGetSymbolAddress                        hipGetSymbolAddress
#define cudaMemcpyKind                              hipMemcpyKind
#define cudaMemcpyHostToDevice                      hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost                      hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice                    hipMemcpyDeviceToDevice
#define cudaMemcpyHostToHost                        hipMemcpyHostToHost

// CUDA exposes a templated T** overload of cudaHostGetDevicePointer, whereas
// HIP only provides the void** form. Provide a matching wrapper (kept as the
// CUDA spelling, so no macro) so typed call sites compile unchanged.
template <typename T>
static inline hipError_t cudaHostGetDevicePointer(T** dev_ptr, void* host_ptr,
                                                  unsigned int flags)
{
    return hipHostGetDevicePointer(reinterpret_cast<void**>(dev_ptr), host_ptr, flags);
}

/* --- device management / properties --- */
#define cudaGetDevice                               hipGetDevice
#define cudaSetDevice                               hipSetDevice
#define cudaGetDeviceCount                          hipGetDeviceCount
#define cudaGetDeviceProperties                     hipGetDeviceProperties
#define cudaDeviceProp                              hipDeviceProp_t

/* --- kernel function attributes (e.g. dynamic shared-memory opt-in) --- */
#define cudaFuncAttributes                          hipFuncAttributes
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize

// CUDA exposes templated T* overloads of cudaFuncSetAttribute /
// cudaFuncGetAttributes that accept a typed kernel function pointer; HIP takes
// const void*. Provide matching wrappers (kept as the CUDA spelling, so no
// macro) so the kernel-pointer call sites compile unchanged.
template <typename T>
static inline hipError_t cudaFuncSetAttribute(T* func, hipFuncAttribute attr, int value)
{
    return hipFuncSetAttribute((const void*)func, attr, value);
}
template <typename T>
static inline hipError_t cudaFuncGetAttributes(hipFuncAttributes* attr, T* func)
{
    return hipFuncGetAttributes(attr, (const void*)func);
}

#endif /* HIP */
#endif /* GPU4PYSCF_HIP_COMPAT_GPU_RUNTIME_H */

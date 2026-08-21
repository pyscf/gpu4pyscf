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
#pragma once

#include <stdio.h>
#include <cuda_runtime.h>

#ifndef USE_SYCL

// copy from samples/common/inc/helper_cuda.h
template <typename T>
int check(T result, char const *const func, const char *const file,
           int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<int>(result), cudaGetErrorName(result), func);
        cudaDeviceReset();
        // Make sure we call CUDA Device Reset before exiting
        //exit(EXIT_FAILURE);
        return 1;
    }
    return 0;
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

#define MALLOC(type, var, size) \
    type *var; \
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&var), sizeof(type) * (size)))
#define FREE(var) \
    checkCudaErrors(cudaFree(var))

#define MEMSET(addr, val, size) \
    checkCudaErrors(cudaMemset(addr, val, size))

#define DEVICE_INIT(type, dst, src, size) \
    MALLOC(type, dst, size); \
    checkCudaErrors(cudaMemcpy(dst, src, sizeof(type) * (size), cudaMemcpyHostToDevice))

#else // !USE_SYCL

// Function to check SYCL errors
template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "SYCL error at " << file << ":" << line << " code=" << result << " \"" << func << "\" \n";
        std::exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(val) (val)


#define MALLOC(type, var, size) \
    type *var = sycl::malloc_device<type>(size, *(sycl_get_queue()));	\
    if (var == nullptr) { \
        std::cerr << "Memory allocation failed for " #var " at " __FILE__ ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    }
    
// Drain the queue before releasing device memory.
//
// The CUDA path uses cudaFree(), which IMPLICITLY SYNCHRONIZES the device, so
// no kernel can still be reading the buffer when it is unmapped. sycl::free()
// has no such guarantee: it unmaps immediately, even with work in flight.
//
// This matters because bpcache pointers (a12/e12/x12/... aliased into the
// single d_aexyz block, bas_coords, bas_atm, bas_pair2bra, ao_loc) are handed
// to the gint kernels INDIRECTLY -- the kernels are launched with
// zeKernelSetIndirectAccess(flags=0x7), so the runtime cannot see them as
// arguments and cannot keep them alive. When GINTdel_basis_prod() runs from
// Python teardown (intopt.clear() / __del__) while int3c1e/int3c2e kernels are
// still executing, the pages are unmapped underneath them:
//
//   zeEventQueryStatus(...) -> ZE_RESULT_NOT_READY   (kernel still running)
//   zeMemFree(0xff000002e8a00000)                    (freed anyway)
//   Segmentation fault from GPU at 0xff000002e8b05000  (base + 0x105000)
//
// The wait restores cudaFree's implicit-sync semantics. These frees are on
// teardown paths, not hot paths, so the cost is negligible.
#define FREE(var) \
    do { \
        sycl_get_queue()->wait(); \
        sycl::free(var, *(sycl_get_queue())); \
    } while (0)

#define MEMSET(addr, val, size) \
    { \
	sycl_get_queue()->submit([&](sycl::handler& cgh) {	\
            cgh.memset(addr, val, size); \
        }).wait(); \
    }

#define DEVICE_INIT(type, dst, src, size) \
    MALLOC(type, dst, size); \
    { \
	sycl_get_queue()->submit([&](sycl::handler& cgh) { \
            cgh.memcpy(dst, src, sizeof(type) * (size)); \
        }).wait(); \
    }

#endif // USE_SYCL

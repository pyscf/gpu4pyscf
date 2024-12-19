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

#include <cuda_runtime.h>

// copy from samples/common/inc/helper_cuda.h
template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<int>(result), cudaGetErrorName(result), func);
        cudaDeviceReset();
        // Make sure we call CUDA Device Reset before exiting
        exit(EXIT_FAILURE);
    }
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


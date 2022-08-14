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


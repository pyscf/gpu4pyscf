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

#include <stdio.h>
#include <cuda_runtime.h>

extern "C" {
__host__
int async_d2h_2d(cudaStream_t stream, double *dst, int dstride, const double *src, int sstride, 
                int rows, int cols)
{
    void* host_ptr = (void *)dst;
    const void* device_ptr = (void *)src;
    int dpitch = dstride;
    int spitch = sstride;
    int width = rows * sizeof(double);
    int height = cols * sizeof(double);
    
    cudaError_t err = cudaMemcpy2DAsync(host_ptr, dpitch, device_ptr, spitch, 
                                        width, height, cudaMemcpyDeviceToHost);
    /*
    cudaError_t err = cudaMemcpy2D(dst, dpitch, src, spitch, 
                                    width, height, cudaMemcpyDeviceToHost);
    */
    printf("%zd \n", sizeof(size_t));
    if(err != cudaSuccess){
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error of d2h_2d\n");
        fprintf(stderr, "err reason %s\n", err_str);
        return 1;

    }
    return 0;
}
}

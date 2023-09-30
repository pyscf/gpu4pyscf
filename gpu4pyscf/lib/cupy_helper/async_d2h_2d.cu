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

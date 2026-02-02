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

#include "gvhf-rys/rys_roots.cu"

static __device__ __forceinline__
void rys_roots_for_k(int nroots, double theta, double rr, double *rw,
                     double omega, double lr_factor, double sr_factor,
                     int block_size, int stride, int rt_id)
{
#ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
#endif
    double theta_rr = theta * rr;
    if (omega == 0) {
        rys_roots(nroots, theta_rr, rw, block_size, rt_id, stride);
        if (lr_factor != 1) {
            __syncthreads();
            for (int irys = rt_id; irys < nroots; irys+=stride) {
                rw[(irys*2+1)*block_size] *= lr_factor;
            }
        }
    } else if (sr_factor == 0) {
        double theta_fac = omega * omega / (omega * omega + theta);
        rys_roots(nroots, theta_fac*theta_rr, rw, block_size, rt_id, stride);
        __syncthreads();
        double sqrt_theta_fac = sqrt(theta_fac) * lr_factor;
        for (int irys = rt_id; irys < nroots; irys+=stride) {
            rw[ irys*2   *block_size] *= theta_fac;
            rw[(irys*2+1)*block_size] *= sqrt_theta_fac;
        }
    } else {
        int _nroots = nroots / 2;
        rys_roots(_nroots, theta_rr, rw, block_size, rt_id, stride);
        double theta_fac = omega * omega / (omega * omega + theta);
        double *rw1 = rw + nroots*block_size;
        rys_roots(_nroots, theta_fac*theta_rr, rw1, block_size, rt_id, stride);
        __syncthreads();
        double full_factor = sr_factor;
        double sqrt_theta_fac = sqrt(theta_fac) * (lr_factor - sr_factor);
        for (int irys = rt_id; irys < _nroots; irys+=stride) {
            rw1[ irys*2   *block_size] *= theta_fac;
            rw1[(irys*2+1)*block_size] *= sqrt_theta_fac;
            rw [(irys*2+1)*block_size] *= full_factor;
        }
    }
}

static __device__ __forceinline__
void rys_roots_for_k(int nroots, double theta, double rr, double *rw,
                     double omega, double lr_factor, double sr_factor)
{
#ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int block_size = item.get_local_range(1);
    int stride = item.get_local_range(0);
    int rt_id = item.get_local_id(0);
#else
    int block_size = blockDim.x;
    int stride = blockDim.y;
    int rt_id = threadIdx.y;
#endif
    rys_roots_for_k(nroots, theta, rr, rw, omega, lr_factor, sr_factor,
                    block_size, stride, rt_id);
}

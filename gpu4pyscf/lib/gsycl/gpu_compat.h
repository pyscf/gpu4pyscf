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

// Backend definitions and abstractions

#pragma once

#include <stddef.h>


#ifdef USE_SYCL

#define setup_context() \
    const auto item = syclex::this_work_item::get_nd_item<3>()

#define threadIdx_x     item.get_local_id(2)
#define threadIdx_y     item.get_local_id(1)
#define threadIdx_z     item.get_local_id(0)

#define blockIdx_x      item.get_group(2)
#define blockIdx_y      item.get_group(1)
#define blockIdx_z      item.get_group(0)

#define blockDim_x      item.get_local_range(2)
#define blockDim_y      item.get_local_range(1)
#define blockDim_z      item.get_local_range(0)

#define gridDim_x       item.get_group_range(2)
#define gridDim_y       item.get_group_range(1)
#define gridDim_z       item.get_group_range(0)

#else

#define setup_context()

#define threadIdx_x     threadIdx.x
#define threadIdx_y     threadIdx.y
#define threadIdx_z     threadIdx.z

#define blockIdx_x      blockIdx.x
#define blockIdx_y      blockIdx.y
#define blockIdx_z      blockIdx.z

#define blockDim_x      blockDim.x
#define blockDim_y      blockDim.y
#define blockDim_z      blockDim.z

#define gridDim_x       gridDim.x
#define gridDim_y       gridDim.y
#define gridDim_z       gridDim.z

#endif


#ifdef USE_SYCL
inline sycl::range<3> make_grid(
    size_t x, size_t y = 1, size_t z = 1)
{
    return sycl::range<3>(z, y, x);
}

inline sycl::range<3> make_block(
    size_t x, size_t y = 1, size_t z = 1)
{
    return sycl::range<3>(z, y, x);
}

#define LAUNCH_KERNEL(kernel, grid, block, shm_size, stream, ...) \
    { \
        (stream).parallel_for( \
            sycl::nd_range<3>(grid * block, block), \
            [=](sycl::nd_item<3>) { kernel(__VA_ARGS__); }); \
    }

#else
inline dim3 make_grid(
    unsigned int x, unsigned int y = 1, unsigned int z = 1)
{
    return dim3(x, y, z);
}

inline dim3 make_block(
    unsigned int x, unsigned int y = 1, unsigned int z = 1)
{
    return dim3(x, y, z);
}

#define LAUNCH_KERNEL(kernel, grid, block, shm_size, stream, ...) \
    { \
        kernel<<<grid, block, shm_size, stream>>>(__VA_ARGS__); \
    }
#endif

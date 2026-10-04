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
// Single header wrapping the most commonly used CUDA constructs so that
// .cu files contain no USE_SYCL ifdefs. Include <cuda_runtime.h> first,
// then this header. The <cuda_runtime.h> shim resolves to the SYCL compat
// layer when USE_SYCL is active, or the real CUDA headers otherwise.

#pragma once

#include <stddef.h>
#include <stdio.h>

// double2 component access. sycl::double2 exposes .x()/.y() methods,
// CUDA double2 exposes .x/.y fields.
#ifdef USE_SYCL
#define D2X(v) ((v).x())
#define D2Y(v) ((v).y())
#else
#define D2X(v) ((v).x)
#define D2Y(v) ((v).y)
#endif

// dim3 shim for auto-generated sources that still spell it (SYCL only).
// Keeps generated files byte-identical with upstream.
#ifdef USE_SYCL
struct gpu4pyscf_dim3 {
    unsigned int x, y, z;
    gpu4pyscf_dim3(unsigned int x=1, unsigned int y=1, unsigned int z=1) : x(x), y(y), z(z) {}
};
#define dim3 gpu4pyscf_dim3
#endif

// Backend-split constant tables with distinct SYCL/CUDA names.
// SYCL keeps a device_global; CUDA uses __constant__. TABLE_BIND aliases
// the active store to the CUDA name so bodies stay backend-agnostic;
// TABLE_FILL publishes host data.
#ifdef USE_SYCL
#define TABLE_DEFINE(type, sname, cname, N) \
    SYCL_EXTERNAL sycl_device_global<type[N]> sname
#define TABLE_BIND(cname, sname) \
    auto cname = sname.get()
#define TABLE_FILL(cname, sname, src, bytes) \
    CONSTANT_MEMCPY(sname, src, bytes)
#else
#define TABLE_DEFINE(type, sname, cname, N) \
    __constant__ type cname[N]
#define TABLE_BIND(cname, sname) \
    (void)0
#define TABLE_FILL(cname, sname, src, bytes) \
    CONSTANT_MEMCPY(cname, src, bytes)
#endif

// Host-pinned buffer address mapping. SYCL uses USM (buffers directly
// accessible from device); CUDA maps the pinned allocation. Call sites
// must return int for the error path.
#ifdef USE_SYCL
#define MAP_PINNED_PTR(type, dev, host) type *dev = (host)
#else
#define MAP_PINNED_PTR(type, dev, host) \
    type *dev; \
    { \
        cudaError_t _err = cudaHostGetDevicePointer(&dev, host, 0); \
        if (_err != cudaSuccess) { \
            fprintf(stderr, "address mapping error %s\n", cudaGetErrorString(_err)); \
            return 1; \
        } \
    }
#endif

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

#define global_x        item.get_global_id(2)
#define global_y        item.get_global_id(1)
#define global_z        item.get_global_id(0)

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

#define global_x        (blockIdx_x * blockDim_x + threadIdx_x)
#define global_y        (blockIdx_y * blockDim_y + threadIdx_y)
#define global_z        (blockIdx_z * blockDim_z + threadIdx_z)

#endif


// Shared / local memory declaration.
// Usage: SHARED_ARRAY(double, tile, [16][16]);
// CUDA: __shared__ double tile[16][16];
// SYCL: group local memory reference bound to `item` from setup_context().
#ifdef USE_SYCL
#define SHARED_ARRAY(type, name, ...) \
    using _smt_##name##_t = type __VA_ARGS__; \
    _smt_##name##_t& name = *sycl::ext::oneapi::group_local_memory_for_overwrite<_smt_##name##_t>(item.get_group())
#else
#define SHARED_ARRAY(type, name, ...) __shared__ type name __VA_ARGS__
#endif

// Dynamically-sized shared / local memory, following the established
// submit+local_accessor pattern. The kernel takes a trailing `void *shm_mem`
// argument in BOTH backends (see LAUNCH_KERNEL_DYN), so kernel signatures
// stay identical. Inside the kernel, after setup_context():
//   DYNAMIC_SHARED_PTR(double, buf, shm_mem);
#ifdef USE_SYCL
#define DYNAMIC_SHARED_PTR(type, name, shm_mem) \
    type *name = static_cast<type *>(shm_mem)
#else
#define DYNAMIC_SHARED_PTR(type, name, shm_mem) \
    extern __shared__ type name[]; \
    (void)(shm_mem)
#endif

// Scalar shared / local variable (one value per work-group, e.g. a block
// counter). Usage: SHARED_SCALAR(int, ntasks); then use `ntasks` as an int.
// CUDA: __shared__ int ntasks;
// SYCL: int-sized group-local slot bound to `item` from setup_context().
#ifdef USE_SYCL
#define SHARED_SCALAR(type, name) \
    using _sms_##name##_t = type[1]; \
    _sms_##name##_t& _sms_##name##_arr = *sycl::ext::oneapi::group_local_memory_for_overwrite<_sms_##name##_t>(item.get_group()); \
    type &name = _sms_##name##_arr[0]
#else
#define SHARED_SCALAR(type, name) __shared__ type name
#endif

// Host-to-constant-memory copy. Usage:
//   CONSTANT_MEMCPY(c_table, h_table, N*sizeof(Entry));
// CUDA: cudaMemcpyToSymbol. SYCL: queue memcpy into the device_global.
#ifdef USE_SYCL
#define CONSTANT_MEMCPY(dst, src, bytes) \
    sycl_get_queue()->memcpy(dst, src, bytes).wait()
#else
#define CONSTANT_MEMCPY(dst, src, bytes) \
    cudaMemcpyToSymbol(dst, src, bytes)
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

#define LAUNCH_KERNEL(kernel, grid, block, shm_size, ...) \
    { \
        sycl_get_queue()->parallel_for( \
            sycl::nd_range<3>(grid * block, block), \
            [=](sycl::nd_item<3>) { kernel(__VA_ARGS__); }); \
    }

// Explicit-stream form for call sites with their own stream/queue object
// or pointer (e.g. solvent, vv10 take cudaStream_t params from callers).
#define LAUNCH_KERNEL_S(kernel, grid, block, shm_size, stream, ...) \
    { \
        _gpu4pyscf_stream_ptr(stream)->parallel_for( \
            sycl::nd_range<3>(grid * block, block), \
            [=](sycl::nd_item<3>) { kernel(__VA_ARGS__); }); \
    }

// Dynamic-shared variant: allocates shm_size bytes of local memory and
// forwards its pointer as a trailing `void *shm_mem` kernel argument in
// BOTH backends (CUDA passes nullptr; the kernel uses `extern __shared__`
// via DYNAMIC_SHARED_PTR instead).
#define LAUNCH_KERNEL_DYN(kernel, grid, block, shm_size, ...) \
    { \
        sycl_get_queue()->submit([&](sycl::handler &cgh) { \
            sycl::local_accessor<char, 1> _dynshm( \
                sycl::range<1>(shm_size), cgh); \
            cgh.parallel_for( \
                sycl::nd_range<3>(grid * block, block), \
                [=](sycl::nd_item<3>) { \
                    kernel(__VA_ARGS__, \
                           GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(_dynshm)); \
                }); \
        }); \
    }

#define LAUNCH_KERNEL_DYN_S(kernel, grid, block, shm_size, stream, ...) \
    { \
        _gpu4pyscf_stream_ptr(stream)->submit([&](sycl::handler &cgh) { \
            sycl::local_accessor<char, 1> _dynshm( \
                sycl::range<1>(shm_size), cgh); \
            cgh.parallel_for( \
                sycl::nd_range<3>(grid * block, block), \
                [=](sycl::nd_item<3>) { \
                    kernel(__VA_ARGS__, \
                           GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(_dynshm)); \
                }); \
        }); \
    }

#else
// Dummy so queue-based launch sites compile in CUDA builds.
// The CUDA LAUNCH_KERNEL macro discards this argument.
static inline void *sycl_get_queue() { return nullptr; }

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

#define LAUNCH_KERNEL(kernel, grid, block, shm_size, ...) \
    { \
        kernel<<<grid, block, shm_size, 0>>>(__VA_ARGS__); \
    }

#define LAUNCH_KERNEL_S(kernel, grid, block, shm_size, stream, ...) \
    { \
        kernel<<<grid, block, shm_size, stream>>>(__VA_ARGS__); \
    }

// Dynamic-shared variant: forwards shm_size as a trailing `void *shm_mem`
// kernel argument in BOTH backends, matching DYNAMIC_SHARED_PTR.
#define LAUNCH_KERNEL_DYN(kernel, grid, block, shm_size, ...) \
    { \
        kernel<<<grid, block, shm_size, 0>>>(__VA_ARGS__, nullptr); \
    }

#define LAUNCH_KERNEL_DYN_S(kernel, grid, block, shm_size, stream, ...) \
    { \
        kernel<<<grid, block, shm_size, stream>>>(__VA_ARGS__, nullptr); \
    }
#endif

// Stream/queue normalization helper: LAUNCH_KERNEL_{S,DYN_S} accept either a
// stream object or a queue pointer in the stream slot, on both backends.
template <typename T> inline T *_gpu4pyscf_stream_ptr(T *s) { return s; }
template <typename T> inline T *_gpu4pyscf_stream_ptr(T &s) { return &s; }

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
#include <tuple>
#include <type_traits>
#include <utility>

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

namespace gpu4pyscf_detail {

// LAUNCH_KERNEL helper machinery. The kernel name arrives as a non-type
// template parameter (C++20): it is a constant expression, so the device
// lambda invokes it directly and SYCL never sees a function pointer.
// Argument handling:
//  * every argument is copied by value into a std::tuple on the host; an
//    argument that is a pointer to a struct (PBCIntEnvVars*, etc.) is
//    dereferenced into the tuple, so `*envs`/`dev_envs` hoists at call
//    sites are unnecessary;
//  * a stream argument is marked with ON_STREAM(stream) in the first
//    kernel-argument slot; anything else goes to the kernel.
struct stream_tag {
    sycl::queue* q;
};
inline stream_tag on_stream(sycl::queue* q) { return {q}; }
inline stream_tag on_stream(sycl::queue& q) { return {&q}; }

template <class Tuple, auto Kernel, std::size_t... I>
inline void _apply_call(const Tuple& t, std::index_sequence<I...>) {
    Kernel(std::get<I>(t)...);
}

template <class Tuple, auto Kernel, std::size_t... I>
inline void _apply_call_shm(const Tuple& t, char* shm, std::index_sequence<I...>) {
    Kernel(std::get<I>(t)..., shm);
}

template <auto Kernel, typename... Args>
inline void launch_submit(sycl::queue* q, sycl::range<3> grid,
                          sycl::range<3> block, Args... args) {
    auto tup = std::make_tuple(args...);
    q->parallel_for(sycl::nd_range<3>(grid * block, block),
        [tup](sycl::nd_item<3>) {
            _apply_call<decltype(tup), Kernel>(tup,
                std::index_sequence_for<Args...>{});
        });
}

template <auto Kernel, typename... Args>
inline void launch_dyn(sycl::queue* q, sycl::range<3> grid,
                       sycl::range<3> block, std::size_t shm_size,
                       Args... args) {
    auto tup = std::make_tuple(args...);
    q->submit([tup, grid, block, shm_size](sycl::handler& cgh) {
        sycl::local_accessor<char, 1> _dynshm(sycl::range<1>(shm_size), cgh);
        cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
            [_dynshm, tup](sycl::nd_item<3>) {
                char* shm = GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(_dynshm);
                _apply_call_shm<decltype(tup), Kernel>(tup, shm,
                    std::index_sequence_for<Args...>{});
            });
    });
}

template <auto Kernel, typename... Args>
inline void launch_kernel(sycl::range<3> grid, sycl::range<3> block,
                          std::size_t shm_size, Args... args) {
    (void)shm_size;
    launch_submit<Kernel>(sycl_get_queue(), grid, block, args...);
}
template <auto Kernel, typename... Args>
inline void launch_kernel(sycl::range<3> grid, sycl::range<3> block,
                          std::size_t shm_size, stream_tag st, Args... args) {
    (void)shm_size;
    launch_submit<Kernel>(st.q, grid, block, args...);
}

template <auto Kernel, typename... Args>
inline void launch_kernel_dyn(sycl::range<3> grid, sycl::range<3> block,
                              std::size_t shm_size, Args... args) {
    launch_dyn<Kernel>(sycl_get_queue(), grid, block, shm_size, args...);
}
template <auto Kernel, typename... Args>
inline void launch_kernel_dyn(sycl::range<3> grid, sycl::range<3> block,
                              std::size_t shm_size, stream_tag st,
                              Args... args) {
    launch_dyn<Kernel>(st.q, grid, block, shm_size, args...);
}

} // namespace gpu4pyscf_detail

#define ON_STREAM(s) gpu4pyscf_detail::on_stream(s)

#define LAUNCH_KERNEL(kernel, grid, block, shm_size, ...) \
    { gpu4pyscf_detail::launch_kernel<kernel>(grid, block, shm_size, __VA_ARGS__); }
#define LAUNCH_KERNEL_DYN(kernel, grid, block, shm_size, ...) \
    { gpu4pyscf_detail::launch_kernel_dyn<kernel>(grid, block, shm_size, __VA_ARGS__); }

#else
// CUDA

// Dummy so sycl_get_queue() call sites compile in CUDA builds.
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

namespace gpu4pyscf_detail {

struct stream_tag {
    cudaStream_t s;
};
inline stream_tag on_stream(cudaStream_t s) { return {s}; }

template <class Kernel, typename... Rest>
inline void launch_kernel(Kernel kernel, dim3 grid, dim3 block,
                          std::size_t shm_size, Rest... rest) {
    kernel<<<grid, block, shm_size, 0>>>(rest...);
}
template <class Kernel, typename... Rest>
inline void launch_kernel(Kernel kernel, dim3 grid, dim3 block,
                          std::size_t shm_size, stream_tag st, Rest... rest) {
    kernel<<<grid, block, shm_size, st.s>>>(rest...);
}
template <class Kernel, typename... Rest>
inline void launch_kernel_dyn(Kernel kernel, dim3 grid, dim3 block,
                              std::size_t shm_size, Rest... rest) {
    kernel<<<grid, block, shm_size, 0>>>(rest..., nullptr);
}
template <class Kernel, typename... Rest>
inline void launch_kernel_dyn(Kernel kernel, dim3 grid, dim3 block,
                              std::size_t shm_size, stream_tag st, Rest... rest) {
    kernel<<<grid, block, shm_size, st.s>>>(rest..., nullptr);
}

} // namespace gpu4pyscf_detail

#define ON_STREAM(s) gpu4pyscf_detail::on_stream(s)

#define LAUNCH_KERNEL(kernel, grid, block, shm_size, ...) \
    { gpu4pyscf_detail::launch_kernel(kernel, grid, block, shm_size, __VA_ARGS__); }
#define LAUNCH_KERNEL_DYN(kernel, grid, block, shm_size, ...) \
    { gpu4pyscf_detail::launch_kernel_dyn(kernel, grid, block, shm_size, __VA_ARGS__); }

#endif

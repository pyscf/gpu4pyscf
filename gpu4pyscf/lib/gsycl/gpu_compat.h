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
#ifdef USE_SYCL
#include <optional>
#include <sycl_device.hpp>
#endif

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
    cudaMemcpyToSymbol(sname, src, bytes)
#else
#define TABLE_DEFINE(type, sname, cname, N) \
    __constant__ type cname[N]
#define TABLE_BIND(cname, sname) \
    (void)0
#define TABLE_FILL(cname, sname, src, bytes) \
    cudaMemcpyToSymbol(cname, src, bytes)
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


// Static shared / local memory declaration, arrays and scalars alike.
// Usage: SHARED_ARRAY(double, tile, [16][16]);  SHARED_ARRAY(int, ntasks);
// CUDA: __shared__ double tile[16][16];          __shared__ int ntasks;
// SYCL: work_group_static storage (sycl_ext_oneapi_work_group_static), not
// group_local_memory: when a TU also uses get_work_group_scratch_memory, DPC++
// gives every group_local_memory kernel an implicit dynamic-local argument,
// set to 0 bytes on scratch-less launches, which OpenCL rejects
// (UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE). Storage is a namespace-scope
// variable template keyed by a function-local tag type: unique per declaring
// function and per template instantiation (a `static` local work_group_static
// is rejected inside function templates). Raw aligned bytes keep
// _for_overwrite semantics (no construction) and allow non-trivially-
// constructible element types such as compat::double3.
#ifdef USE_SYCL
namespace gpu4pyscf_detail {
template <typename T> struct _wgs_raw { alignas(T) unsigned char bytes[sizeof(T)]; };
template <typename T, typename Tag>
sycl::ext::oneapi::experimental::work_group_static<_wgs_raw<T>> _wgs_storage;
}
#define SHARED_ARRAY(type, name, ...) \
    using _smt_##name##_t = type __VA_ARGS__; \
    struct _wgs_tag_##name {}; \
    _smt_##name##_t& name = *reinterpret_cast<_smt_##name##_t*>( \
        (&gpu4pyscf_detail::_wgs_storage<_smt_##name##_t, _wgs_tag_##name>)->bytes)
#else
#define SHARED_ARRAY(type, name, ...) __shared__ type name __VA_ARGS__
#endif

// Dynamically-sized shared / local memory. SYCL uses the
// sycl_ext_oneapi_work_group_scratch_memory extension: the kernel fetches a
// base pointer itself via get_work_group_scratch_memory() and the launch
// communicates the byte size through the work_group_scratch_size property
// (see launch_kernel below, via nd_launch + launch_config). No explicit
// pointer is passed through the kernel signature on the SYCL side.
// Static shared uses sycl_ext_oneapi_work_group_static storage (see
// SHARED_ARRAY above), likewise with no kernel arg.
// CUDA keeps the extern __shared__ idiom, so kernel signatures stay
// identical with no shared-memory argument on either side.
// Inside the kernel, after setup_context():
//   DYNAMIC_SHARED_PTR(double, buf);
// An optional trailing argument (e.g. a legacy `shm_mem` pointer) is
// tolerated and ignored on both backends
// (CUDA uses extern __shared__, SYCL uses scratch memory).
#ifdef USE_SYCL
#define DYNAMIC_SHARED_PTR(type, name, ...) \
    type *name = static_cast<type *>( \
        sycl::ext::oneapi::experimental::get_work_group_scratch_memory())
#else
#define DYNAMIC_SHARED_PTR(type, name, ...) \
    extern __shared__ type name[]
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
//  * every argument is copied by value into a std::tuple on the host (so
//    `*envs`-style dereferences are evaluated at the call site);
//  * a stream/queue argument (`sycl::queue*` / `sycl::queue&`) anywhere in
//    the pack selects the launch stream and is removed from the kernel call.
//    No ON_STREAM() marker needed; kernels never legitimately take a
//    queue as a parameter.
template <typename T>
struct _is_stream
    : std::bool_constant<
          std::is_same_v<std::decay_t<T>, sycl::queue> ||
          std::is_same_v<std::decay_t<T>, sycl::queue*>> {};

inline sycl::queue* _stream_ptr(sycl::queue& r) { return &r; }
inline sycl::queue* _stream_ptr(sycl::queue* p) { return p; }

template <typename... Args>
inline sycl::queue* _pick_stream(Args&&... args) {
    sycl::queue* q = sycl_get_queue();
    auto scan = [&](auto&& v) { if constexpr (_is_stream<decltype(v)>::value) q = _stream_ptr(v); };
    (scan(args), ...);
    return q;
}

template <class Tuple, std::size_t... I>
inline auto _filter_streams(const Tuple& t, std::index_sequence<I...>) {
    return std::tuple_cat(
        [ ] (const auto& v) {
            using E = std::remove_cv_t<std::remove_reference_t<decltype(v)>>;
            if constexpr (_is_stream<E>::value) return std::tuple{};
            else return std::tuple{v};
        }(std::get<I>(t)) ... );
}

template <class Tuple, auto Kernel, std::size_t... I>
inline void _apply_call(const Tuple& t, std::index_sequence<I...>) {
    Kernel(std::get<I>(t)...);
}

template <class Tuple, auto Kernel, std::size_t... I>
inline void _apply_call_shm(const Tuple& t, char* shm, std::index_sequence<I...>) {
    Kernel(std::get<I>(t)..., shm);
}

// Compile-time detection: does the kernel accept a trailing dynamic shared
// memory pointer? Kernels declare no such argument (DYNAMIC_SHARED_PTR
// fetches the scratch allocation itself), but if one ever does, it gets a
// nullptr placeholder (unused by the body) while the bytes still come from
// the work_group_scratch_size launch property. Either way no local_accessor
// / explicit device pointer is created.
template <auto Kernel, class Tuple, std::size_t... I>
consteval bool _kernel_takes_shm(std::index_sequence<I...>) {
    return requires(char* shm) { Kernel(std::get<I>(std::declval<Tuple>())..., shm); };
}

// Record a rejected launch so cudaGetLastError() reports it at the call site.
inline void _launch_error(const char *what, const char *kernel) {
    snprintf(gpu4pyscf_last_launch_error_msg,
             sizeof(gpu4pyscf_last_launch_error_msg), "%s [%s]", what, kernel);
    gpu4pyscf_last_launch_error = cudaErrorInvalidConfiguration;
    fprintf(stderr, "LAUNCH_KERNEL error: %s\n", gpu4pyscf_last_launch_error_msg);
}

// shm_size is the dynamic shared memory in BYTES (CUDA's third <<<>>>
// argument): pass n * sizeof(T), never an element count. Rejects sizes the
// device cannot provide; the per-device limit is cached per thread.
inline bool _shm_bytes_ok(sycl::queue &q, std::size_t shm_bytes, const char *kernel) {
    if (shm_bytes == 0) return true;
    static thread_local std::optional<sycl::device> dev;
    static thread_local std::size_t limit = 0;
    sycl::device d = q.get_device();
    if (!dev || !(*dev == d)) {
        dev = d;
        limit = d.get_info<sycl::info::device::local_mem_size>();
    }
    if (shm_bytes <= limit) return true;
    char msg[256];
    snprintf(msg, sizeof(msg),
             "dynamic shared memory of %zu bytes exceeds the %zu-byte work-group "
             "limit (shm_size is in BYTES: n*sizeof(T))", shm_bytes, limit);
    _launch_error(msg, kernel);
    return false;
}

template <auto Kernel, typename ShmT, typename... Args>
inline void launch_kernel(sycl::range<3> grid,
                          sycl::range<3> block, ShmT shm_arg,
                          Args... args) {
    static_assert(std::is_integral_v<ShmT>,
                  "LAUNCH_KERNEL: shm_size must be an integer byte count");
    if constexpr (std::is_signed_v<ShmT>) {
        if (shm_arg < 0) {
            _launch_error("negative dynamic shared memory size", __PRETTY_FUNCTION__);
            return;
        }
    }
    const std::size_t shm_size = static_cast<std::size_t>(shm_arg);
    namespace syclexp = sycl::ext::oneapi::experimental;
    sycl::queue* q = _pick_stream(args...);
    if (!_shm_bytes_ok(*q, shm_size, __PRETTY_FUNCTION__)) return;
    auto tup_all = std::make_tuple(args...);
    auto tup = _filter_streams(tup_all, std::index_sequence_for<Args...>{});
    constexpr bool with_shm =
        _kernel_takes_shm<Kernel, decltype(tup)>(
            std::make_index_sequence<std::tuple_size<decltype(tup)>::value>{});
    try {
        const sycl::nd_range<3> ndr(grid * block, block);
        if constexpr (with_shm) {
            // shm_size is in BYTES, matching the CUDA <<<..., shm_bytes, ...>>>
            // convention. Legacy trailing pointer receives nullptr; the kernel
            // body aliases scratch memory via DYNAMIC_SHARED_PTR.
            syclexp::launch_config cfg{
                ndr, syclexp::properties{syclexp::work_group_scratch_size{shm_size}}};
            syclexp::nd_launch(*q, cfg,
                [tup](sycl::nd_item<3>) {
                    _apply_call_shm<decltype(tup), Kernel>(tup, static_cast<char*>(nullptr),
                        std::make_index_sequence<std::tuple_size<decltype(tup)>::value>{});
                });
        } else {
            if (shm_size == 0) {
                q->parallel_for(ndr,
                    [tup](sycl::nd_item<3>) {
                        _apply_call<decltype(tup), Kernel>(tup,
                            std::make_index_sequence<std::tuple_size<decltype(tup)>::value>{});
                    });
            } else {
                syclexp::launch_config cfg{
                    ndr, syclexp::properties{syclexp::work_group_scratch_size{shm_size}}};
                syclexp::nd_launch(*q, cfg,
                    [tup](sycl::nd_item<3>) {
                        _apply_call<decltype(tup), Kernel>(tup,
                            std::make_index_sequence<std::tuple_size<decltype(tup)>::value>{});
                    });
            }
        }
    } catch (const sycl::exception &e) {
        _launch_error(e.what(), __PRETTY_FUNCTION__);
    }
}

} // namespace gpu4pyscf_detail

#define LAUNCH_KERNEL(kernel, grid, block, shm_size, ...) \
    { gpu4pyscf_detail::launch_kernel<kernel>(grid, block, shm_size, __VA_ARGS__); }

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

// Auto-detect stream argument anywhere in the pack (kernel args never carry
// a real stream, so the trait-vs-value match is unambiguous).
template <typename T>
struct _is_stream
    : std::bool_constant<std::is_same_v<std::decay_t<T>, cudaStream_t>> {};

inline cudaStream_t _stream_ptr(cudaStream_t s) { return s; }

template <typename... Args>
inline cudaStream_t _pick_stream(Args&&... args) {
    cudaStream_t s = 0;
    auto scan = [&](auto&& v) { if constexpr (_is_stream<decltype(v)>::value) s = _stream_ptr(v); };
    (scan(args), ...);
    return s;
}

template <class Tuple, std::size_t... I>
inline auto _filter_streams(const Tuple& t, std::index_sequence<I...>) {
    return std::tuple_cat(
        [ ] (const auto& v) {
            using E = std::remove_cv_t<std::remove_reference_t<decltype(v)>>;
            if constexpr (_is_stream<E>::value) return std::tuple{};
            else return std::tuple{v};
        }(std::get<I>(t)) ... );
}

// shm_size is the dynamic shared memory in BYTES: pass n * sizeof(T).
template <class Kernel, typename ShmT, typename... Args>
inline void launch_kernel(Kernel kernel, dim3 grid, dim3 block,
                          ShmT shm_arg, Args... args) {
    static_assert(std::is_integral_v<ShmT>,
                  "LAUNCH_KERNEL: shm_size must be an integer byte count");
    if constexpr (std::is_signed_v<ShmT>) {
        if (shm_arg < 0) {
            fprintf(stderr, "LAUNCH_KERNEL error: negative dynamic shared memory size [%s]\n",
                    __PRETTY_FUNCTION__);
            return;
        }
    }
    const std::size_t shm_size = static_cast<std::size_t>(shm_arg);
    cudaStream_t s = _pick_stream(args...);
    auto tup = _filter_streams(std::make_tuple(args...), std::index_sequence_for<Args...>{});
    std::apply([&](auto... a) {
        if constexpr (std::is_invocable_v<Kernel, decltype(a)..., void*>) {
            // Kernel takes a trailing dynamic shared memory pointer.
            kernel<<<grid, block, shm_size, s>>>(a..., nullptr);
        } else {
            kernel<<<grid, block, shm_size, s>>>(a...);
        }
    }, tup);
}

} // namespace gpu4pyscf_detail

#define LAUNCH_KERNEL(kernel, grid, block, shm_size, ...) \
    { gpu4pyscf_detail::launch_kernel(kernel, grid, block, shm_size, __VA_ARGS__); }

#endif

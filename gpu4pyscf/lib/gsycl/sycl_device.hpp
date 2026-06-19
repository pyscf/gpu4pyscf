#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <stdexcept>

#include <sycl/sycl.hpp>

#define warpSize (item.get_sub_group().get_max_local_range()[0])

using cudaError_t = int;
constexpr int cudaSuccess = 0;
inline unsigned int __activemask() { return 0; }
inline int cudaGetLastError() { return 0; }
inline int cudaPeekAtLastError() { return 0; }
inline void checkCudaErrors(int) { }
inline const char* cudaGetErrorString(int) { return "No error"; }

#define cudaGetDevice(ptr) (syclGetDevice(ptr))

#if defined(__SYCL_DEVICE_ONLY__)
#define printf(...) sycl::ext::oneapi::experimental::printf(__VA_ARGS__)
#endif

extern "C" {
  SYCL_EXTERNAL unsigned __attribute__((overloadable)) intel_get_slice_id(void);
  SYCL_EXTERNAL unsigned __attribute__((overloadable)) intel_get_subslice_id(void);
  SYCL_EXTERNAL unsigned __attribute__((overloadable)) intel_get_eu_id(void);
}
inline void __trap() { __builtin_trap(); }

enum cudaFuncAttribute {
  cudaFuncAttributeMaxDynamicSharedMemorySize = 0,
  cudaFuncAttributePreferredSharedMemoryCarveout = 1,
};

enum cudaFuncCache {
  cudaFuncCachePreferNone   = 0,
  cudaFuncCachePreferShared = 1,
  cudaFuncCachePreferL1     = 2,
  cudaFuncCachePreferEqual  = 3
};

#ifndef cudaFuncSetAttribute
#define cudaFuncSetAttribute(...) (cudaSuccess)
#endif

#ifndef cudaFuncSetCacheConfig
#define cudaFuncSetCacheConfig(...) (cudaSuccess)
#endif

#define CUDA_VERSION 12040
#define __maxnreg__(x)

#define __global__ __attribute__((always_inline))
#define __device__ __attribute__((always_inline))
#define __forceinline__ __attribute__((always_inline))
#define __host__ __attribute__((always_inline))
#define __constant__ inline constexpr

using cudaStream_t = sycl::queue&;
namespace syclex = sycl::ext::oneapi;
using double2 = sycl::double2;

#define rnorm3d(d1,d2,d3) (1 / sycl::length(sycl::double3((d1), (d2), (d3))))
#define norm3d(d1,d2,d3) (sycl::length(sycl::double3((d1), (d2), (d3))))
#define __syncthreads() (sycl::group_barrier(item.get_group()))
#define __syncwarps() (sycl::group_barrier(item.get_sub_group()))
#define __threadfence_block() (sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::work_group))
#define __shfl_down_sync(mask, val, delta) (sycl::shift_group_left((item.get_sub_group()), (val), (delta)))
#define __ballot_sync(mask, predicate) (sycl::reduce_over_group(item.get_sub_group(), (predicate) ? (0x1 << item.get_sub_group().get_local_linear_id()) : 0, sycl::plus<>()))
#define __popc(x) (sycl::popcount(x))

template <typename T> __attribute__((always_inline)) auto ceil(T x) { return sycl::ceil(x); }
template <typename T> __attribute__((always_inline)) auto sqrtf(T x) { return sycl::sqrt(x); }
template <typename T> __attribute__((always_inline)) auto sqrt(T x) { return sycl::sqrt(x); }
template <typename T> __attribute__((always_inline)) auto rsqrt(T x) { return sycl::rsqrt(x); }
template <typename T> __attribute__((always_inline)) auto min(T x, T y) { return sycl::min(x, y); }
template <typename T> __attribute__((always_inline)) auto max(T x, T y) { return sycl::max(x, y); }
template <typename T> __attribute__((always_inline)) auto exp(T x) { return sycl::exp(x); }
template <typename T> __attribute__((always_inline)) auto expf(T x) { return sycl::exp(x); }
template <typename T> __attribute__((always_inline)) auto fabs(T x) -> std::enable_if_t<std::is_same_v<T, double>, double> { return sycl::fabs(x); }
template <typename T> __attribute__((always_inline)) auto fabsf(T x) -> std::enable_if_t<std::is_same_v<T, float>, float> { return sycl::fabs(x); }
template <typename T> __attribute__((always_inline)) auto erf(T x) { return sycl::erf(x); }
template <typename T> __attribute__((always_inline)) auto floor(T x) { return sycl::floor(x); }
template <typename T> __attribute__((always_inline)) auto pow(T x, int n) { return sycl::pown(x, n); }
template <typename T, typename U> __attribute__((always_inline)) auto pow(T x, U n) { return sycl::pow(x, n); }
template <typename T> __attribute__((always_inline)) typename std::enable_if<std::is_same<T, float>::value, float>::type logf(T x) { return sycl::log(x); }
template <typename T> __attribute__((always_inline)) auto log(T x) { return sycl::log(x); }
template <typename T> __attribute__((always_inline)) void sincos(T x, T* sptr, T* cptr) { *sptr = sycl::sincos(x, cptr); }
#define NAN std::numeric_limits<float>::quiet_NaN()

namespace constants {
    constexpr double pi = 3.141592653589793238462643383279502884;
}
#ifndef M_PI
#define M_PI constants::pi
#endif

namespace compat {
  struct double3 {
    double x, y, z;
    constexpr double3() : x(0.0), y(0.0), z(0.0) {}
    constexpr double3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
  };
}
using double3 = compat::double3;

template <typename T1, typename T2>
static inline T1
atomicAdd(T1* addr, const T2 val) {
    sycl::atomic_ref<T1,
        sycl::memory_order::acq_rel,
        sycl::memory_scope::device,
        sycl::access::address_space::generic_space> atom(*(addr));
    return atom.fetch_add(static_cast<T1>((val)));
}
template <typename T1, typename T2>
static inline T1
atomicMax(T1* addr, const T2 val) {
    sycl::atomic_ref<T1,
        sycl::memory_order::relaxed,
        sycl::memory_scope::device,
        sycl::access::address_space::generic_space> atom(*(addr));
    return atom.fetch_max(static_cast<T1>((val)));
}
template <typename T>
static inline typename std::enable_if<std::is_integral<T>::value, T>::type
atomicOr(T* addr, const T val) {
    sycl::atomic_ref<T,
        sycl::memory_order::relaxed,
        sycl::memory_scope::device,
        sycl::access::address_space::generic_space> atom(*(addr));
    return atom.fetch_or((val));
}

template <class T>
using sycl_device_global = sycl::ext::oneapi::experimental::device_global<T>;

#if defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER >= 20230200
#define GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(accessor) \
  accessor.get_multi_ptr<sycl::access::decorated::no>().get()
#else
#define GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(accessor) accessor.get_pointer()
#endif

// =====================================================================
// Queue management — DECLARATIONS ONLY
//
// Implementation lives in sycl_api_python.cpp, compiled into ONE .so
// (libgsycl.so).  All other .so files (libgdft, libgint, ...) link
// against libgsycl.so so the process has exactly ONE sycl::queue* per
// device.
//
// Ownership model: Python owns the queues.  On `import cupy`,
// cupy/cuda.py creates one in-order dpctl.SyclQueue per GPU and calls
// sycl_set_queue_ptr(device_id, q.addressof_ref()) for each.  C++ never
// creates queues and never enumerates devices — Python is authoritative.
//
// Contract for C++ callers: sycl_get_queue_ptr() will throw if called
// before Python has registered a queue for the current device.  In
// practice this means any C++ path reachable from Python is safe, since
// `import cupy` runs first; standalone C++ test binaries must call
// sycl_set_queue_ptr() themselves.
//
// Thread safety: g_current_device is std::atomic — any thread can read
// the current device id.  sycl_set_device() / sycl_set_queue_ptr() are
// serialised by an internal mutex.
// =====================================================================

extern "C" {
  int   sycl_get_device_id();
  void* sycl_get_queue_ptr();
  void* sycl_get_queue_ptr_for(int device_id);
  void  sycl_set_queue_ptr(int device_id, void* queue_ptr);
  void  sycl_set_device(int device_id);
}

// Convenience: return typed sycl::queue* for C++ callers
static inline sycl::queue* sycl_get_queue() {
  return static_cast<sycl::queue*>(sycl_get_queue_ptr());
}

static inline int syclGetDevice(int* id) {
  *id = sycl_get_device_id();
  return 0;
}

static inline void syclSetDevice(int id) {
  sycl_set_device(id);
}

// --- CUDA-compat device-property helpers ---

struct cudaDeviceProp {
  int multiProcessorCount;
};

static inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int /*device*/) {
  prop->multiProcessorCount = static_cast<int>(
    sycl_get_queue()->get_device().get_info<sycl::info::device::max_compute_units>());
  return cudaSuccess;
}

// --- CUDA-compat memory helpers (use current thread's queue) ---

static inline void cudaMalloc(void** ptr, size_t size) {
  (*ptr) = sycl::malloc_device(size, *(sycl_get_queue()));
}

static inline void cudaFree(void* ptr) {
  sycl::free(ptr, *(sycl_get_queue()));
}

static inline void cudaMemset(void* ptr, int val, size_t size) {
  sycl_get_queue()->memset(ptr, static_cast<unsigned char>(val), size).wait();
}

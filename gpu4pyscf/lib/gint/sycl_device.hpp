#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>
#include <type_traits>
//#include <numbers> C++20 feature for value of PI

#include <sys/syscall.h>
#include <unistd.h>

#include <sycl/sycl.hpp>

#define warpSize (item.get_sub_group().get_max_local_range()[0]) // (SYCL builtin not available) Needed for the files: ./lib/pbc/./int3c2e_create_tasks.cuh, ./int3c2e_create_tasks_o1.cuh, ./lib/pbc/unrolled_int3c2e.cu

using cudaError_t = int;
constexpr int cudaSuccess = 0;
inline unsigned int __activemask() { return 0; }
inline int cudaGetLastError() { return 0; }
inline int cudaPeekAtLastError() { return 0; }
inline void checkCudaErrors(int) { }
inline const char* cudaGetErrorString(int) { return "No error"; }

#define cudaGetDevice(ptr) (syclGetDevice(ptr))

/*** CUDA-kernel printf → SYCL device printf mapping ***/
#if defined(__SYCL_DEVICE_ONLY__)
// Redirect any device-side printf(...) to SYCL's experimental printf(...)
#define printf(...) sycl::ext::oneapi::experimental::printf(__VA_ARGS__)
#endif

extern "C" {
  SYCL_EXTERNAL unsigned __attribute__((overloadable)) intel_get_slice_id(void);
  SYCL_EXTERNAL unsigned __attribute__((overloadable)) intel_get_subslice_id(void);
  SYCL_EXTERNAL unsigned __attribute__((overloadable)) intel_get_eu_id(void);
}
inline void __trap() { __builtin_trap(); }

// Function attributes (only what your code needs)
enum cudaFuncAttribute {
  cudaFuncAttributeMaxDynamicSharedMemorySize = 0,
  cudaFuncAttributePreferredSharedMemoryCarveout = 1,
  // add others here if your code references them
};

// Cache preference enum (in case your code uses it)
enum cudaFuncCache {
  cudaFuncCachePreferNone   = 0,
  cudaFuncCachePreferShared = 1,
  cudaFuncCachePreferL1     = 2,
  cudaFuncCachePreferEqual  = 3
};


// Return-value no-ops
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
//#define __syncthreads() (item.barrier())
#define __syncthreads() (sycl::group_barrier(item.get_group()))
#define __syncwarps() (sycl::group_barrier(item.get_sub_group()))
#define __threadfence_block() (sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::work_group))
#define __shfl_down_sync(mask, val, delta) (sycl::shift_group_left((item.get_sub_group()), (val), (delta)))
#define __ballot_sync(mask, predicate) (sycl::reduce_over_group(item.get_sub_group(), (predicate) ? (0x1 << item.get_sub_group().get_local_linear_id()) : 0, sycl::plus<>()))
#define __popc(x) (sycl::popcount(x))

template <typename T> __attribute__((always_inline)) auto ceil(T x) { return sycl::ceil(x); }
template <typename T> __attribute__((always_inline)) auto sqrtf(T x) { return sycl::sqrt(x); }
template <typename T> __attribute__((always_inline)) auto sqrt(T x) { return sycl::sqrt(x); }
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
// Only define M_PI if not already defined (to avoid conflict)
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

// #ifdef SYCL_EXT_ONEAPI_DEVICE_GLOBAL
template <class T>
using sycl_device_global = sycl::ext::oneapi::experimental::device_global<T>;
// #else
// template <class T>
// using sycl_device_global = sycl::ext::oneapi::experimental::device_global<
//     T,
//     decltype(sycl::ext::oneapi::experimental::properties(
//         sycl::ext::oneapi::experimental::device_image_scope))>;
// #endif


#if defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER >= 20230200
#define GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(accessor) \
  accessor.get_multi_ptr<sycl::access::decorated::no>().get()
#else
#define GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(accessor) accessor.get_pointer()
#endif


auto asyncHandler = [](sycl::exception_list exceptions) {
  for (std::exception_ptr const &e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (sycl::exception const &e) {
      std::cerr << "Caught asynchronous SYCL exception:" << std::endl
		<< e.what() << std::endl
		<< "Exception caught at file:" << __FILE__
		<< ", line:" << __LINE__ << std::endl;
    }
  }
};

class device_ext: public sycl::device {
public:
  device_ext(): sycl::device() {}
  ~device_ext() { std::lock_guard<std::mutex> lock(m_mutex); }
  device_ext(const sycl::device& base): sycl::device(base) {}

private:
  mutable std::mutex m_mutex;
};

static inline int get_tid() { return syscall(SYS_gettid); }

class dev_mgr {
public:
  int current_device() {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto tid = get_tid();
    auto it = _thread2dev_map.find(tid);
    if(it != _thread2dev_map.end()) {
        check_id(it->second);
        return it->second;
    }
    // Insert default device if not present
    _thread2dev_map[tid] = DEFAULT_DEVICE_ID;
    return DEFAULT_DEVICE_ID;
  }
  sycl::queue* current_queue() {
    return _queues[current_device()];
  }
  sycl::queue* select_queue(int id) {
    std::lock_guard<std::mutex> lock(m_mutex);
    check_id(id);
    return _queues[id];
  }
  void select_device(int id) {
    std::lock_guard<std::mutex> lock(m_mutex);
    check_id(id);
    _thread2dev_map[get_tid()] = id;
  }
  int device_count() { return _queues.size(); }

  /// Returns the instance of device manager singleton.
  static dev_mgr& instance() {
    static dev_mgr d_m{};
    return d_m;
  }
  dev_mgr(const dev_mgr&)            = delete;
  dev_mgr& operator=(const dev_mgr&) = delete;
  dev_mgr(dev_mgr&&)                 = delete;
  dev_mgr& operator=(dev_mgr&&)      = delete;

private:
  mutable std::mutex m_mutex;

  dev_mgr() {
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devices.empty()) {
      throw std::runtime_error("No SYCL GPU devices found.");
    }

    for (const auto& dev : devices) {
      auto* q = new sycl::queue(dev, asyncHandler, sycl::property_list{sycl::property::queue::in_order{}});
      _queues.push_back(q);
    }
    // sycl::device dev{sycl::gpu_selector_v};
    // _queues.push_back(new sycl::queue(dev, asyncHandler, sycl::property_list{sycl::property::queue::in_order{}}));
  }

  void check_id(int id) const {
    if(id >= _queues.size()) { throw std::runtime_error("Invalid device id"); }
  }

  std::vector<sycl::queue*> _queues;

  /// DEFAULT_DEVICE_ID is used, if current_device() can not find current
  /// thread id in _thread2dev_map, which means default device should be used
  /// for the current thread.
  const int DEFAULT_DEVICE_ID = 0;
  /// thread-id to device-id map.
  std::map<int, int> _thread2dev_map;
};

/// Util function to get the current device (in int).
static inline int syclGetDevice(int* id) { *id = dev_mgr::instance().current_device(); return 0; }

/// Util function to get the current queue
static inline sycl::queue* sycl_get_queue() {
  return dev_mgr::instance().current_queue();
}
/// Util function to get queue from device`id`
static inline sycl::queue* sycl_get_queue_nth(int device_id) {
  return dev_mgr::instance().select_queue(device_id);
}


/// Util function to set a device by id. (to _thread2dev_map)
static inline void syclSetDevice(int id) { dev_mgr::instance().select_device(id); }

/// Util function to get number of GPU devices (default: explicit scaling)
static inline void syclGetDeviceCount(int* id) { *id = dev_mgr::instance().device_count(); }

static inline void cudaMalloc(void** ptr, size_t size) {
  (*ptr) = sycl::malloc_device(size, *(sycl_get_queue()));
}
static inline void cudaFree(void* ptr) {
  sycl::free(ptr, *(sycl_get_queue()));
}
static inline void cudaMemset(void* ptr, int val, size_t size) {
  sycl_get_queue()->memset(ptr, static_cast<unsigned char>(val), size).wait();
}

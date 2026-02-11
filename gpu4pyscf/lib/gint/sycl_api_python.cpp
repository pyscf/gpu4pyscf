#include "sycl_device.hpp"
#include <unordered_map>
#include <mutex>

static std::unordered_map<void*, std::shared_ptr<sycl::event>> event_map;
static std::mutex event_mutex;

// Ensure exported symbols are visible when building the shared library
#define GPU4PYSCF_EXPORT __attribute__((visibility("default")))

extern "C" {

GPU4PYSCF_EXPORT int sycl_get_device_id() {
  int id{0};
  syclGetDevice(&id);
  return id;
}

GPU4PYSCF_EXPORT void* sycl_get_queue_ptr() {
  return static_cast<void*>(sycl_get_queue());
}

GPU4PYSCF_EXPORT void* sycl_get_queue_ptr_nth(int device_id) {
  return static_cast<void*>(sycl_get_queue_nth(device_id));
}

GPU4PYSCF_EXPORT void sycl_set_device(int device_id) {
  syclSetDevice(device_id);
}

GPU4PYSCF_EXPORT int sycl_get_device_count() {
  int count{0};
  syclGetDeviceCount(&count);
  return count;
}

GPU4PYSCF_EXPORT void sycl_queue_synchronize(void* queue_ptr) {
  auto* q = static_cast<sycl::queue*>(queue_ptr);
  q->wait();  // Wait for all enqueued operations to finish
}

GPU4PYSCF_EXPORT void* sycl_record_event() {
  auto queue = sycl_get_queue();
  auto ev = std::make_shared<sycl::event>(queue->ext_oneapi_submit_barrier());

  void* handle = static_cast<void*>(ev.get());
  {
    std::lock_guard<std::mutex> lock(event_mutex);
    event_map[handle] = ev;
  }

  return handle;
}

GPU4PYSCF_EXPORT void sycl_wait_event(void* handle) {
  std::shared_ptr<sycl::event> ev;
  {
    std::lock_guard<std::mutex> lock(event_mutex);
    auto it = event_map.find(handle);
    if (it != event_map.end()) {
      ev = it->second;
      event_map.erase(it);  // optional: clean up
    }
  }

  if (ev) {
    ev->wait();
  }
}

GPU4PYSCF_EXPORT size_t sycl_get_total_memory() {
  auto dev = sycl_get_queue()->get_device();
  return dev.get_info<sycl::info::device::global_mem_size>();
}
GPU4PYSCF_EXPORT size_t sycl_get_shared_memory() {
  auto dev = sycl_get_queue()->get_device();
  return dev.get_info<sycl::info::device::local_mem_size>();
}

GPU4PYSCF_EXPORT size_t sycl_get_free_memory() {
  auto dev = sycl_get_queue()->get_device();
  if (!dev.has(sycl::aspect::ext_intel_free_memory)) {
    std::cout << "Device " << dev.get_info<sycl::info::device::name>()
              << " does not support ext_intel_free_memory." << std::endl;
    return (0.9 * dev.get_info<sycl::info::device::global_mem_size>());
  }
  return dev.get_info<sycl::ext::intel::info::device::free_memory>();
}

// // Optional: CUDA-like memset helper
// GPU4PYSCF_EXPORT void sycl_cuda_memset(void* ptr, int value, size_t size) {
//   cudaMemset(ptr, value, size);
// }

GPU4PYSCF_EXPORT size_t sycl_memcpy(void* dst, void* src, size_t size) {
  sycl_get_queue()->memcpy(dst, src, size);
}

}

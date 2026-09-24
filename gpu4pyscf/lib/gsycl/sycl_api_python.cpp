#include <string>
#include "sycl_device.hpp"

#define GPU4PYSCF_EXPORT __attribute__((visibility("default")))

// =====================================================================
// Process-global queue storage.
//
// This is the ONE place queues live. All .so files that include
// sycl_device.hpp call the extern "C" functions below (resolved at
// link/load time to THIS translation unit in libgsycl.so).
//
// Python creates in-order dpctl.SyclQueue objects and pushes them
// here via sycl_set_queue_ptr(). No queues are created in C++.
// =====================================================================

static std::mutex                g_mutex;
static std::vector<sycl::queue*> g_queues;          // one per GPU, set by Python
static std::atomic<int>          g_current_device{0};     // process-global

// =====================================================================
// Event storage (for sycl_record_event / sycl_wait_event)
// =====================================================================

static std::unordered_map<void*, std::shared_ptr<sycl::event>> g_event_map;
static std::mutex g_event_mutex;

// =====================================================================
// Exported C API — called by ALL .so files through the header
// =====================================================================

extern "C" {

GPU4PYSCF_EXPORT int sycl_get_device_id() {
    return g_current_device.load(std::memory_order_relaxed);
}

GPU4PYSCF_EXPORT void* sycl_get_queue_ptr() {
    int id = g_current_device.load(std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(g_mutex);
    if (id < 0 || id >= static_cast<int>(g_queues.size()) ||
        g_queues[id] == nullptr) {
        throw std::runtime_error(
            "sycl_get_queue_ptr: queue for device " + std::to_string(id) +
            " not set. Ensure 'import cupy' runs before any GPU op.");
    }
    return static_cast<void*>(g_queues[id]);
}
GPU4PYSCF_EXPORT void* sycl_get_queue_ptr_for(int device_id) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (device_id < 0 ||
        device_id >= static_cast<int>(g_queues.size()) ||
        g_queues[device_id] == nullptr) {
        throw std::runtime_error(
            "sycl_get_queue_ptr_for: no queue registered for device " +
            std::to_string(device_id));
    }
    return static_cast<void*>(g_queues[device_id]);
}
  
GPU4PYSCF_EXPORT void sycl_set_queue_ptr(int device_id, void* queue_ptr) {
    if (device_id < 0)
        throw std::runtime_error("sycl_set_queue_ptr: negative device id");
    std::lock_guard<std::mutex> lock(g_mutex);
    if (static_cast<size_t>(device_id) >= g_queues.size())
        g_queues.resize(device_id + 1, nullptr);
    g_queues[device_id] = static_cast<sycl::queue*>(queue_ptr);
}

GPU4PYSCF_EXPORT void sycl_set_device(int device_id) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (device_id < 0 ||
        device_id >= static_cast<int>(g_queues.size()) ||
        g_queues[device_id] == nullptr) {
        throw std::runtime_error(
            "sycl_set_device: no queue registered for device " +
            std::to_string(device_id));
    }
    g_current_device.store(device_id, std::memory_order_relaxed);
}

// GPU4PYSCF_EXPORT void sycl_queue_synchronize(void* queue_ptr) {
//     auto* q = static_cast<sycl::queue*>(queue_ptr);
//     q->wait();
// }

// GPU4PYSCF_EXPORT void* sycl_record_event() {
//     auto* q = static_cast<sycl::queue*>(sycl_get_queue_ptr());
//     auto ev = std::make_shared<sycl::event>(q->ext_oneapi_submit_barrier());

//     void* handle = static_cast<void*>(ev.get());
//     {
//         std::lock_guard<std::mutex> lock(g_event_mutex);
//         g_event_map[handle] = ev;
//     }
//     return handle;
// }

// GPU4PYSCF_EXPORT void sycl_wait_event(void* handle) {
//     std::shared_ptr<sycl::event> ev;
//     {
//         std::lock_guard<std::mutex> lock(g_event_mutex);
//         auto it = g_event_map.find(handle);
//         if (it != g_event_map.end()) {
//             ev = it->second;
//             g_event_map.erase(it);
//         }
//     }
//     if (ev) {
//         try {
//             ev->wait();
//         } catch (...) {
//             // Swallow — may happen during shutdown
//         }
//     }
// }

GPU4PYSCF_EXPORT size_t sycl_get_total_memory() {
    auto* q = static_cast<sycl::queue*>(sycl_get_queue_ptr());
    return q->get_device().get_info<sycl::info::device::global_mem_size>();
}

GPU4PYSCF_EXPORT size_t sycl_get_shared_memory() {
    auto* q = static_cast<sycl::queue*>(sycl_get_queue_ptr());
    return q->get_device().get_info<sycl::info::device::local_mem_size>();
}

// Maps to CUDA cudaDeviceProp::multiProcessorCount
GPU4PYSCF_EXPORT int sycl_get_compute_units() {
    auto* q = static_cast<sycl::queue*>(sycl_get_queue_ptr());
    return static_cast<int>(
        q->get_device().get_info<sycl::info::device::max_compute_units>());
}

// Maps to CUDA cudaDeviceProp::name. Copies the device name into the
// caller-provided buffer (NUL-terminated, truncated to buf_size-1).
GPU4PYSCF_EXPORT void sycl_get_device_name(char* buf, int buf_size) {
    if (buf == nullptr || buf_size <= 0) return;
    auto* q = static_cast<sycl::queue*>(sycl_get_queue_ptr());
    std::string name = q->get_device().get_info<sycl::info::device::name>();
    int n = static_cast<int>(name.size());
    if (n > buf_size - 1) n = buf_size - 1;
    for (int i = 0; i < n; ++i) buf[i] = name[i];
    buf[n] = '\0';
}

GPU4PYSCF_EXPORT size_t sycl_get_free_memory() {
    auto* q = static_cast<sycl::queue*>(sycl_get_queue_ptr());
    auto dev = q->get_device();
    if (!dev.has(sycl::aspect::ext_intel_free_memory)) {
        return static_cast<size_t>(
            0.9 * dev.get_info<sycl::info::device::global_mem_size>());
    }
    return dev.get_info<sycl::ext::intel::info::device::free_memory>();
}

GPU4PYSCF_EXPORT size_t sycl_memcpy(void* dst, void* src, size_t size) {
    auto* q = static_cast<sycl::queue*>(sycl_get_queue_ptr());
    q->memcpy(dst, src, size);
    return 0;
}

} // extern "C"

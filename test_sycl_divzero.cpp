#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    sycl::queue q{ sycl::gpu_selector_v };

    constexpr size_t N = 4;
    // allocate a small array on device
    float *data = sycl::malloc_device<float>(N, q);

    // initialize array to some values (optional)
    q.memset(data, 0, N * sizeof(float)).wait();

    // Launch kernel that intentionally divides by zero
    q.parallel_for<class div0_demo>(sycl::range<1>(N), [=](sycl::id<1> idx) {
      float x = 1.0f;
      float y = 0.0f;      // divisor = 0
      float z = x / y;     // <-- UB
      data[idx] = z;       // store result to observe
    }).wait();

    // read back results
    float host_buf[N];
    q.memcpy(host_buf, data, N * sizeof(float)).wait();

    std::cout << "Results after division-by-zero kernel:\n";
    for (size_t i = 0; i < N; ++i) {
        std::cout << "host_buf[" << i << "] = " << host_buf[i] << "\n";
    }

    sycl::free(data, q);
    return 0;
}

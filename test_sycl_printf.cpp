#include <sycl/sycl.hpp>

// #ifdef __SYCL_DEVICE_ONLY__
// #define __SYCL_CONSTANT_AS __attribute__((opencl_constant))
// #else
// #define __SYCL_CONSTANT_AS
// #endif

// const __SYCL_CONSTANT_AS char fmt[] = "Hello, World! %f\n";

int main() {
  sycl::queue q;

  q.submit([&](sycl::handler &cgh) {
    cgh.single_task([=]() {
      float f = 3.14;
      sycl::ext::oneapi::experimental::printf("%f\n", f);
    });
  });

  return 0;
}

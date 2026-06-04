#pragma once
// Drop-in replacement for <cuda_runtime.h> when building with SYCL.
// Source files include <cuda_runtime.h> unconditionally — this file
// is found first on the include path when USE_SYCL is active.
#include "sycl_device.hpp"

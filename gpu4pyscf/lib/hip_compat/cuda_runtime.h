/* CUDA -> HIP compatibility stub. See gpu_runtime.h in this directory.
 * Lets existing `#include <cuda_runtime.h>` lines resolve on ROCm builds.
 * This directory is placed on the include path only for HIP builds. */
#include "gpu_runtime.h"

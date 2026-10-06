# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# INTERFACE target — carries the include path to any consumer

# ── gsycl: SYCL compatibility layer ────────────────────────────────────
#
# Provides two things:
#   1. sycl_compat  — INTERFACE target that injects gsycl/ at the front
#                     of the include path, making <cuda_runtime.h> and
#                     <cuda.h> resolve to the SYCL shim headers instead
#                     of the real CUDA toolkit headers.
#   2. gsycl        — shared library exposing the Python ABI (extern "C"
#                     device/queue management symbols) via libgsycl.so.
#                     All other .so files link against this to resolve
#                     sycl_get_queue_ptr, sycl_set_queue_ptr, etc.

# 1. Header-only INTERFACE target — consumed by every other subdirectory
add_library(sycl_compat INTERFACE)
target_include_directories(sycl_compat INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
  "$<INSTALL_INTERFACE:include/gsycl>"
)

# 2. Python ABI shared library — ONE translation unit owns all queue state
add_library(gsycl SHARED sycl_api_python.cpp)
target_link_libraries(gsycl PRIVATE sycl_compat)
set_target_properties(gsycl PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

# 3. Namespaced alias — lets consumers write gsycl::gsycl instead of gsycl,
#    making it immediately clear this is an external dependency target
add_library(gsycl::gsycl ALIAS gsycl)

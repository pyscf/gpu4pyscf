# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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

set(GPU_SRCS
  nr_ecp_driver.cu
)

add_library(gecp SHARED ${GPU_SRCS})

if (USE_SYCL)
  set_source_files_properties(${GPU_SRCS} PROPERTIES LANGUAGE CXX)
  # Use COMPILE_FLAGS for reliable per-target flags
  set_target_properties(gecp PROPERTIES
    COMPILE_FLAGS "-fsycl -x c++ -nocudainc -nocudalib -O2"
  )
  # ocloc tuning for the 21MB per_kernel-split SPV emitted by this TU:
  #  * -fsycl-max-parallel-link-jobs=N raises llvm-foreach --jobs (default 4)
  #  * a second -Xsycl-target-backend appends "-options ..." to EVERY ocloc
  #    invocation (both the bundle compile and the per-kernel llvm-foreach
  #    step). The whole backend string must be single-quoted for the shell
  #    with the -options payload in inner double quotes, so icpx forwards
  #    "-ze-opt-level=1" to ocloc as ONE argument. Without
  #    the inner quotes icpx splits the words and ocloc rejects
  #    "-ze-opt-level=1" as a top-level option (Invalid option, exit 226).
  #    -cl-opt-disable was REMOVED (2026-10-14): with IGC optimization
  #    passes disabled, the ECP ipip kernels are miscompiled into
  #    cumulative private-scratch corruption -> GPU "Write (NotPresent)"
  #    aborts in unrelated later kernels (see progress.md). With opts
  #    enabled the lib is correct; link cost rises ~30 min -> ~2h25m.
  #    The historical IGC SIGSEGV (error -11) did NOT reappear on the
  #    2026.1 stack. Options concatenate with the parent's global
  #    "-device pvc" flag (duplicate -device is harmless).
  set_target_properties(gecp PROPERTIES
    LINK_FLAGS "-fsycl-max-parallel-link-jobs=32 -Xsycl-target-backend=spir64_gen '-device pvc -options \"-ze-opt-level=1\"'"
  )
  target_link_libraries(gecp PRIVATE sycl_compat gsycl)
else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")# -maxrregcount=128")

  # If compiling with CUDA<13.1 and arch>100, ECP will silently provide incorrect result.
  set(HAS_BLACKWELL_OR_NEWER OFF)
  foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX MATCH "([0-9]+)" arch_number ${arch})
    if("${arch_number}" STREQUAL "")
      message(FATAL_ERROR "Cannot extract architecture id from CMAKE_CUDA_ARCHITECTURES term ${arch}.")
    endif()
    if(arch_number GREATER_EQUAL 100)
      set(HAS_BLACKWELL_OR_NEWER ON)
      break()
    endif()
  endforeach()

  set(CUDA_VERSION_LESS_THAN_13_1 OFF)
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "13.1")
    set(CUDA_VERSION_LESS_THAN_13_1 ON)
  endif()

  if(HAS_BLACKWELL_OR_NEWER AND CUDA_VERSION_LESS_THAN_13_1)
    message(WARNING "Blackwell or newer GPU detected with sm ${arch_number} >= 100, and a CUDA version ${CMAKE_CUDA_COMPILER_VERSION} < 13.1 is detected. "
                    "To get around with a severe bug in nvcc, we turn off compiler optimization. The performance will be damaged.")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Ofast-compile max")
  endif()
  # End of hacking CUDA<13.1 and arch>100
endif()

set_target_properties(gecp PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
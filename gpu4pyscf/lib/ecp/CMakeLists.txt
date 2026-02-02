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

set(GPU_SRCS nr_ecp_driver.cu)

add_library(gecp SHARED ${GPU_SRCS})

if (USE_SYCL)
  file(GLOB CUH_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh")
  set_source_files_properties(${CUH_HEADERS} PROPERTIES LANGUAGE CXX)
  set_source_files_properties(${GPU_SRCS} PROPERTIES LANGUAGE CXX)
  target_compile_options(gecp PRIVATE -x c++ -nocudainc -nocudalib)
else ()
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

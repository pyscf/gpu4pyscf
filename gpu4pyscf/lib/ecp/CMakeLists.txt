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

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")# -maxrregcount=128")

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

if(HAS_BLACKWELL_OR_NEWER)
  message(WARNING "Blackwell or newer GPU detected with sm ${arch_number} >= 100. To get around with a severe bug in nvcc, we turn off compiler optimization. The performance will be damaged.")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Ofast-compile max")
endif()

add_library(gecp SHARED
  nr_ecp_driver.cu
)

set_target_properties(gecp PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

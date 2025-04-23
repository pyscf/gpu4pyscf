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
endif()

set_target_properties(gecp PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

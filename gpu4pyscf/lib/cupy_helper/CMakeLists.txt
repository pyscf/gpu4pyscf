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

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")

set(cupy_helper_src
  transpose.cu
  block_diag.cu
  unpack.cu
  take_last2d.cu
  async_d2h_2d.cu
  add_sparse.cu
  dist_matrix.cu
  cart2sph.cu
)

if(BUILD_CUTLASS)
  set(cupy_helper_src ${cupy_helper_src}
    grouped_gemm.cu
    grouped_dot.cu
  )
endif()

add_library(cupy_helper SHARED ${cupy_helper_src})

if(BUILD_CUTLASS)
add_dependencies(cupy_helper cutlass)
target_include_directories(cupy_helper PRIVATE ${cutlass_SOURCE_DIR}/tools/util/include ${cutlass_SOURCE_DIR}/include)
endif()

set_target_properties(cupy_helper PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
set_target_properties(cupy_helper PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

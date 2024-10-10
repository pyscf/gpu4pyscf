# gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
#
# Copyright (C) 2022 Qiming Sun
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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

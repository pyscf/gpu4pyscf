# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
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

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_30 -rdc=true")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60 --ptxas-options=-v")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_50 --ptxas-options=-v -maxrregcount=255")

set(CMAKE_VERBOSE_MAKEFILE ON)
add_library(gint SHARED 
  g2e.c
  pair_data.c
  bpcache.cu
  constant.cu
  nr_fill_ao_ints.cu
  nr_fill_ao_int3c2e.cu
  nr_fill_ao_int3c2e_ip1.cu
  nr_fill_ao_int3c2e_ip2.cu
  nr_fill_ao_int3c2e_ipip1.cu
  nr_fill_ao_int3c2e_ipip2.cu
  nr_fill_ao_int3c2e_ip1ip2.cu
  nr_fill_ao_int3c2e_ipvip1.cu
)

#option(BUILD_SHARED_LIBS "build shared libraries" 1)
#option(ENABLE_STATIC "Enforce static library build" 0)
#if(ENABLE_STATIC)
#  set(BUILD_SHARED_LIBS 0)
#endif()

set_target_properties(gint PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  CUDA_SEPARABLE_COMPILATION ON)

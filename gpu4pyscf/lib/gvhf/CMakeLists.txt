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

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v -maxrregcount=128")

add_library(gvhf SHARED
  constant.cu
  #nr_jk_driver.cu
  nr_jk_driver_int3c2e_ip1.cu
  nr_jk_driver_int3c2e_ip2.cu
  nr_jk_driver_int3c2e_pass1.cu
  nr_jk_driver_int3c2e_pass2.cu
  #nr_jk_driver_ip1.cu
  #get_veff_driver_ip1.cu
  rys_roots_dat.cu
)

#option(BUILD_SHARED_LIBS "build shared libraries" 1)
#option(ENABLE_STATIC "Enforce static library build" 0)
#if(ENABLE_STATIC)
#  set(BUILD_SHARED_LIBS 0)
#endif()

set_target_properties(gvhf PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(gvhf gint)

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

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v -maxrregcount=128")

add_library(gvhf SHARED
  constant.cu
  nr_jk_driver.cu
  nr_jk_driver_int3c2e_ip1.cu
  nr_jk_driver_int3c2e_ip2.cu
  nr_jk_driver_int3c2e_pass1.cu
  nr_jk_driver_int3c2e_pass2.cu
  nr_jk_driver_ip1.cu
        get_veff_driver_ip1.cu
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

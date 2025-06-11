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

add_library(dpnp_helper SHARED onemkl_lapack.cpp)

set_target_properties(dpnp_helper PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)

# Find Intel oneMKL
find_package(MKL REQUIRED CONFIG PATHS "$ENV{MKLROOT}")

#target_compile_options(dpnp_helper PRIVATE -qmkl)

# Link MKL SYCL interface
target_link_libraries(dpnp_helper PRIVATE MKL::MKL_SYCL)

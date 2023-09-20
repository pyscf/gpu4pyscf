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

add_library(cupy_helper SHARED 
  transpose.cu
  block_diag.cu
  contract_cderi_k.cu
  eigh.cu
  cho_solve.cu
  take_last2d.cu
  async_d2h_2d.cu
  cutensor.cu
  add_sparse.cu
)

set_target_properties(cupy_helper PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
#find_package(CUDAToolkit)


if (NOT DEFINED ENV{CUTENSOR_ROOT} AND NOT DEFINED CUTENSOR_ROOT)
    message(FATAL_ERROR "cuTensor not found. CUTENSOR_ROOT must be defined.")
else ()
    if (DEFINED ENV{CUTENSOR_ROOT})
        set(CUTENSOR_ROOT "$ENV{CUTENSOR_ROOT}")
    endif ()
    message("-- Looking for cuTENSOR in ${CUTENSOR_ROOT}")
    if (NOT EXISTS ${CUTENSOR_ROOT})
        message("Cannot find CUTENSOR_ROOT")
    else ()
        add_definitions(-DUSE_CUTENSOR)

        if (NOT TARGET cutensor)
            add_library(cutensor SHARED IMPORTED)
            if (WIN32)
                set(CUTENSOR_LIBRARY_NAME "cutensor.dll")
                set(CUTENSOR_LIBRARY_DEF "cutensor.lib")
            else ()
                set(CUTENSOR_LIBRARY_NAME "libcutensor.so.1")
            endif ()
            set(LIB_DIR "/lib")

            set_target_properties(cutensor PROPERTIES
                    IMPORTED_LOCATION "${CUTENSOR_ROOT}/${LIB_DIR}/${CUTENSOR_LIBRARY_NAME}"
                    IMPORTED_IMPLIB "${CUTENSOR_ROOT}/${LIB_DIR}/${CUTENSOR_LIBRARY_DEF}"
                    INTERFACE_INCLUDE_DIRECTORIES "${CUTENSOR_ROOT}/include")
        endif ()

        include_directories(SYSTEM ${CUDA_INCLUDE_DIRS} ${CUTENSOR_ROOT}/include)

        message("-- cuTENSOR linked.")

    endif ()
endif ()

find_package(NVHPC)
if(${NVHPC_FOUND})
  message("looks like it's using NVHPC.")
  target_link_libraries(cupy_helper PRIVATE NVHPC::CUBLAS PRIVATE NVHPC::CUSOLVER PRIVATE NVHPC::CUTENSOR)
else()
  target_link_libraries(cupy_helper PRIVATE CUDA::cublas PRIVATE CUDA::cusolver PRIVATE cutensor)
endif()


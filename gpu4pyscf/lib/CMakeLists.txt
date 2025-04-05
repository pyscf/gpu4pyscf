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

cmake_minimum_required (VERSION 3.19 FATAL_ERROR) # 3.19 is required by cutlass
project (gpu4pyscf C CXX CUDA Fortran)

set(CMAKE_C_STANDARD "99")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# For better performance on A100, the option
# -DCUDA_ARCHITECTURES="70-real;80-real" can be added to cmake command
if(DEFINED CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")
else()
  set(CMAKE_CUDA_ARCHITECTURES "70-real;80-real;90-real")
endif()
message("CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
endif()
#set(CMAKE_BUILD_TYPE DEBUG)
message("Build type ${CMAKE_BUILD_TYPE}")

set(CMAKE_VERBOSE_MAKEFILE OFF)
if (CMAKE_COMPILER_IS_GNUCC) # Does it skip the link flag on old OsX?
  # TODO: If updating to minimum requirement cmake>=3.7, use
  # CMAKE_SHARED_LINKER_FLAGS_INIT to combine LDFLAGS options.
  # https://cmake.org/cmake/help/v3.7/variable/CMAKE_SHARED_LINKER_FLAGS_INIT.html
  # see also issue #661
  if(UNIX AND NOT APPLE AND NOT DEFINED ENV{LDFLAGS})
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed")
  endif()
endif()
set(CMAKE_C_FLAGS "-Wall ${CMAKE_C_FLAGS}")

option(BUILD_MARCH_NATIVE "gcc flag -march=native" off)
if (BUILD_MARCH_NATIVE)
  include(CheckCCompilerFlag)
  CHECK_C_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
  if(COMPILER_SUPPORTS_MARCH_NATIVE)
    if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
      message("Add CFLAGS -march=native -unroll-aggressive")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -unroll-aggressive -ipo")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-math-errno")
    else()
      message("Add CFLAGS -march=native -ftree-vectorize")
      #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -O2")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -funroll-loops -ftree-vectorize")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math -fno-math-errno -fno-strict-overflow")
    endif()
  endif()
endif()

option(ENABLE_OPENMP "Compiling C extensions with openmp" ON)
set(OPENMP_C_PROPERTIES "")
if(ENABLE_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(HAVE_OPENMP 1)
    set(OPENMP_C_PROPERTIES OpenMP::OpenMP_C)
  endif()
endif()

# See also https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling
if (WIN32)
  #?
elseif (APPLE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/deps/lib;@loader_path/deps/lib64")
  set(CMAKE_BUILD_RPATH "@loader_path;@loader_path/deps/lib;@loader_path/deps/lib64")
else ()
  set(CMAKE_SKIP_BUILD_RPATH  True)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH True)
  set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/deps/lib:\$ORIGIN/deps/lib64")
endif ()

#if(BLKSIZE)
#  message("Block size for integral buffer ${BLKSIZE}")
#  add_definitions(-DBLKSIZE=${BLKSIZE})
#endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/deps/include)
link_directories(${PROJECT_SOURCE_DIR}/deps/lib)

#list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

set(C_LINK_TEMPLATE "<CMAKE_C_COMPILER> <CMAKE_SHARED_LIBRARY_C_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
set(CXX_LINK_TEMPLATE "<CMAKE_CXX_COMPILER> <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")

option(BUILD_CUTLASS "Using cutlass" ON)
if(BUILD_CUTLASS)
  include(ExternalProject)
  ExternalProject_Add(cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG        v3.4.0
    CONFIGURE_COMMAND ""  # No configure step
    BUILD_COMMAND ""      # No build step
    INSTALL_COMMAND ""    # No install step
    TEST_COMMAND ""       # No test step
  )

  # ExternalProject_Add automatically populates this variable
  ExternalProject_Get_Property(cutlass SOURCE_DIR)
  set(cutlass_SOURCE_DIR ${SOURCE_DIR})
endif()

option(BUILD_GINT "Using gint" ON)
if(BUILD_GINT)
  add_subdirectory(gint)
endif()

option(BUILD_GVHF "Using gvhf" ON)
if(BUILD_GVHF)
  add_subdirectory(gvhf)
endif()

option(BUILD_GDFT "Using gdft" ON)
if(BUILD_GDFT)
  add_subdirectory(gdft)
endif()

option(BUILD_CUPY_HELPER "Using cupy_helper" ON)
if(BUILD_CUPY_HELPER)
  add_subdirectory(cupy_helper)
endif()

option(BUILD_SOLVENT "Using SMD solvent" ON)
if(BUILD_SOLVENT)
  add_subdirectory(solvent)
endif()


option(BUILD_GINT_RYS "Using gint-rys" ON)
if(BUILD_GINT_RYS)
  add_subdirectory(gint-rys)
endif()

option(BUILD_GVHF_RYS "Using gvhf-rys" ON)
if(BUILD_GVHF_RYS)
  add_subdirectory(gvhf-rys)
endif()

option(BUILD_GVHF_MD "Using gvhf-md" ON)
if(BUILD_GVHF_MD)
  add_subdirectory(gvhf-md)
endif()

option(BUILD_PBC "Using pbc" ON)
if(BUILD_PBC)
  add_subdirectory(pbc)
endif()

add_subdirectory(multigrid)
add_subdirectory(ecp)

option(BUILD_LIBXC "Using libxc for DFT" ON)
if(BUILD_LIBXC)
  include(ExternalProject)
  ExternalProject_Add(libxc
    GIT_REPOSITORY https://github.com/wxj6000/libxc.git
    GIT_TAG b225c254c063e1de835a4425115c9a6377478b32
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    CMAKE_ARGS -DBUILD_SHARED_LIBS=ON -DENABLE_CUDA=ON
            -DENABLE_FORTRAN=OFF -DDISABLE_KXC=OFF -DDISABLE_LXC=ON -DDISABLE_FHC=ON
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
            -DCMAKE_INSTALL_LIBDIR:PATH=lib
            -DCMAKE_C_CREATE_SHARED_LIBRARY=${C_LINK_TEMPLATE}
            -DENABLE_XHOST:STRING=${BUILD_MARCH_NATIVE}
    CMAKE_CACHE_ARGS -DCMAKE_CUDA_ARCHITECTURES:STRING=${CMAKE_CUDA_ARCHITECTURES}
  )
endif()

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
option(USE_SYCL "Using SYCL backend" ON)
if (USE_SYCL)
  project (gpu4pyscf C CXX Fortran)
else()
  project (gpu4pyscf C CXX CUDA Fortran)
endif()

set(CMAKE_C_STANDARD "99")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

<<<<<<< HEAD
if (USE_SYCL)
find_package(IntelSYCL)
endif()

if(USE_SYCL)
add_definitions(-DUSE_SYCL=1)
# add_definitions(-DSYCL_EXT_ONEAPI_DEVICE_GLOBAL=1)
endif(USE_SYCL)

if (NOT USE_SYCL)
  # For better performance on A100, the option
  # -DCUDA_ARCHITECTURES="70-real;80-real" can be added to cmake command
  if(DEFINED CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")
  else()
    set(CMAKE_CUDA_ARCHITECTURES "60-real;70-real;80-real;90-real")
  endif()
  message("CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
=======
# For better performance on A100, the option
# -DCUDA_ARCHITECTURES="70-real;80-real" can be added to cmake command
if(DEFINED CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")
else()
  set(CMAKE_CUDA_ARCHITECTURES "70-real;80-real;90-real")
>>>>>>> origin/master
endif()

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
endif()
#set(CMAKE_BUILD_TYPE DEBUG)
message("Build type ${CMAKE_BUILD_TYPE}")

set(CMAKE_VERBOSE_MAKEFILE ON) #vama temp ON
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

# option(BUILD_MARCH_NATIVE "gcc flag -march=native" off)
# if (BUILD_MARCH_NATIVE)
#   include(CheckCCompilerFlag)
#   CHECK_C_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
#   if(COMPILER_SUPPORTS_MARCH_NATIVE)
#     if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
#       message("Add CFLAGS -march=native -unroll-aggressive")
#       set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -unroll-aggressive -ipo")
#       set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-math-errno")
#     else()
#       message("Add CFLAGS -march=native -ftree-vectorize")
#       #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -O2")
#       set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -funroll-loops -ftree-vectorize")
#       set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math -fno-math-errno -fno-strict-overflow")
#     endif()
#   endif()
# endif()

# # See also https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling
# if (WIN32)
#   #?
# elseif (APPLE)
#   set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
#   set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/deps/lib;@loader_path/deps/lib64")
#   set(CMAKE_BUILD_RPATH "@loader_path;@loader_path/deps/lib;@loader_path/deps/lib64")
# else ()
#   set(CMAKE_SKIP_BUILD_RPATH  True)
#   set(CMAKE_BUILD_WITH_INSTALL_RPATH True)
#   set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/deps/lib:\$ORIGIN/deps/lib64")
# endif ()

# #if(BLKSIZE)
# #  message("Block size for integral buffer ${BLKSIZE}")
# #  add_definitions(-DBLKSIZE=${BLKSIZE})
# #endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/deps/include)
link_directories(${PROJECT_SOURCE_DIR}/deps/lib)

# #list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

set(DPCTL_CMAKE_MODULES_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${DPCTL_CMAKE_MODULES_PATH})
find_package(IntelSYCL REQUIRED PATHS ${DPCTL_CMAKE_MODULES_PATH} NO_DEFAULT_PATH)

set(C_LINK_TEMPLATE "<CMAKE_C_COMPILER> <CMAKE_SHARED_LIBRARY_C_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
set(CXX_LINK_TEMPLATE "<CMAKE_CXX_COMPILER> <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")

# option(BUILD_CUTLASS "Using cutlass" OFF) #vama off
# if(BUILD_CUTLASS)
#   include(ExternalProject)
#   ExternalProject_Add(cutlass
#     GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
#     GIT_TAG        v3.4.0
#     CONFIGURE_COMMAND ""  # No configure step
#     BUILD_COMMAND ""      # No build step
#     INSTALL_COMMAND ""    # No install step
#     TEST_COMMAND ""       # No test step
#   )

#   # ExternalProject_Add automatically populates this variable
#   ExternalProject_Get_Property(cutlass SOURCE_DIR)
#   set(cutlass_SOURCE_DIR ${SOURCE_DIR})
# endif()

add_subdirectory(gint)
add_subdirectory(gvhf)
add_subdirectory(gdft)
if (USE_SYCL)
  add_subdirectory(dpnp_helper)
else()
  add_subdirectory(cupy_helper)
endif()
add_subdirectory(solvent)

# option(BUILD_LIBXC "Using libxc for DFT" OFF) #vama temporal
# if(BUILD_LIBXC)
#   include(ExternalProject)
#   ExternalProject_Add(libxc
#     GIT_REPOSITORY https://gitlab.com/libxc/libxc.git
#     GIT_TAG 6af8da52125a05e5997fb65751eee7fe4bc2f171
#     PREFIX ${PROJECT_BINARY_DIR}/deps
#     INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
#     CMAKE_ARGS -DBUILD_SHARED_LIBS=ON -DENABLE_CUDA=ON
#             -DENABLE_FORTRAN=OFF -DDISABLE_KXC=ON -DDISABLE_LXC=ON -DDISABLE_FHC=ON
#             -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
#             -DCMAKE_INSTALL_LIBDIR:PATH=lib
#             -DCMAKE_C_CREATE_SHARED_LIBRARY=${C_LINK_TEMPLATE}
#             -DENABLE_XHOST:STRING=${BUILD_MARCH_NATIVE}
#     CMAKE_CACHE_ARGS -DCMAKE_CUDA_ARCHITECTURES:STRING=${CMAKE_CUDA_ARCHITECTURES}
#   )
# endif()

<<<<<<< HEAD
# # ---- compilation for dftd3 and dft4
# # 1. build static dependencies
# # 2. build dftd3 and dftd4 shared libs, dftd3 and dftd4 will automatically search their dependencies
# # https://github.com/dftd4/dftd4/blob/3fc00439c6abea2639868b644c52f0920d6c2e22/config/cmake/Findmstore.cmake#L24
# option(BUILD_DFTD3 "Using DFTD3 for DFT" ON)
# if(BUILD_DFTD3)
#   include(ExternalProject)
#   ExternalProject_Add(dftd3_static
#     GIT_REPOSITORY "https://github.com/dftd3/simple-dftd3"
#     GIT_TAG v1.0.0
#     PREFIX ${PROJECT_BINARY_DIR}/deps
#     INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
#     CMAKE_ARGS -DWITH_OpenMP=OFF
#             -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
#             -DCMAKE_INSTALL_LIBDIR:PATH=lib
#             -DCMAKE_CURRENT_SOURCE_DIR:PATH=dftd3
#   )

#   include(ExternalProject)
#   ExternalProject_Add(dftd3
#     GIT_REPOSITORY "https://github.com/dftd3/simple-dftd3"
#     GIT_TAG v1.0.0
#     PREFIX ${PROJECT_BINARY_DIR}/deps
#     INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
#     CMAKE_ARGS -DWITH_OpenMP=OFF -DBUILD_SHARED_LIBS=ON
#             -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
#             -DCMAKE_INSTALL_LIBDIR:PATH=lib
#             -DCMAKE_CURRENT_SOURCE_DIR:PATH=dftd3
#   )
#   add_dependencies(dftd3 dftd3_static)
# endif()

# option(BUILD_DFTD4 "Using DFTD4 for DFT" ON)
# if(BUILD_DFTD4)
#   include(ExternalProject)
#   ExternalProject_Add(dftd4_static
#     GIT_REPOSITORY "https://github.com/dftd4/dftd4"
#     GIT_TAG v3.6.0
#     PREFIX ${PROJECT_BINARY_DIR}/deps
#     INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
#     CMAKE_ARGS -DWITH_OpenMP=OFF
#             -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
#             -DCMAKE_INSTALL_LIBDIR:PATH=lib
#   )

#   include(ExternalProject)
#   ExternalProject_Add(dftd4
#     GIT_REPOSITORY "https://github.com/dftd4/dftd4"
#     GIT_TAG v3.6.0
#     PREFIX ${PROJECT_BINARY_DIR}/deps
#     INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
#     CMAKE_ARGS -DWITH_OpenMP=OFF -DBUILD_SHARED_LIBS=ON
#             -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
#             -DCMAKE_INSTALL_LIBDIR:PATH=lib
#   )
#   add_dependencies(dftd4 dftd4_static)
# endif()
=======
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
>>>>>>> origin/master

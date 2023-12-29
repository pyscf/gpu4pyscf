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

# 3.18 for CUDAToolkit and CMAKE_CUDA_ARCHITECTURES
cmake_minimum_required (VERSION 3.18)
project (gpu4pyscf C CUDA)

set(CMAKE_C_STANDARD "99")

# For better performance on A100, the option
# -DCUDA_ARCHITECTURES="70-real;80-real" can be added to cmake command
if(DEFINED CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")
else()
  set(CMAKE_CUDA_ARCHITECTURES "70-real;80-real")
endif()

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

add_subdirectory(gint)
add_subdirectory(gvhf)
add_subdirectory(gdft)
add_subdirectory(cupy_helper)

option(BUILD_LIBXC "Using libxc for DFT" ON)
if(BUILD_LIBXC)
  include(ExternalProject)
  ExternalProject_Add(libxc
    GIT_REPOSITORY https://gitlab.com/libxc/libxc.git
    GIT_TAG 6af8da52125a05e5997fb65751eee7fe4bc2f171
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    CMAKE_ARGS -DBUILD_SHARED_LIBS=ON -DENABLE_CUDA=ON
            -DENABLE_FORTRAN=OFF -DDISABLE_KXC=ON -DDISABLE_LXC=ON -DDISABLE_FHC=ON
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
            -DCMAKE_INSTALL_LIBDIR:PATH=lib
            -DCMAKE_C_CREATE_SHARED_LIBRARY=${C_LINK_TEMPLATE}
            -DENABLE_XHOST:STRING=${BUILD_MARCH_NATIVE}
    CMAKE_CACHE_ARGS -DCMAKE_CUDA_ARCHITECTURES:STRING=${CMAKE_CUDA_ARCHITECTURES}
  )
endif()

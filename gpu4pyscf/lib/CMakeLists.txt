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

if (USE_SYCL)
  set(CMAKE_BUILT_TYPE Debug)
  set(DPCTL_CMAKE_MODULES_PATH "${PROJECT_SOURCE_DIR}/cmake")
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${DPCTL_CMAKE_MODULES_PATH})
  #find_package(IntelSYCL REQUIRED PATHS ${DPCTL_CMAKE_MODULES_PATH} NO_DEFAULT_PATH)

  add_definitions(-DUSE_SYCL=1)
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -Wno-register -Wsycl-strict -fsycl-device-code-split=per_kernel -sycl-std=2020 -fp-model=precise -fsycl-max-parallel-link-jobs=20 -mllvm -enable-global-offset=false ")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -Wno-register -fsycl-device-code-split=per_kernel -fp-model=precise -fsycl-max-parallel-link-jobs=20 -mllvm -enable-global-offset=false -fsycl-default-sub-group-size 32 -fno-system-debug -flink-huge-device-code --offload-compress -fsycl-max-parallel-link-jobs=16") # -fsanitize=signed-integer-overflow -fno-sanitize-recover=signed-integer-overflow -fsanitize=address")
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen '-device pvc' -fsycl-device-code-split=per_kernel")

  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -Wno-register -Wsycl-strict -fsycl-device-code-split=per_kernel -fsycl-targets=intel_gpu_pvc -sycl-std=2020 -fp-model=precise -fsycl-max-parallel-link-jobs=20 -mllvm -enable-global-offset=false -fsycl-default-sub-group-size 32 -fno-system-debug -flink-huge-device-code --offload-compress -fsycl-max-parallel-link-jobs=16") # -fsanitize=signed-integer-overflow -fno-sanitize-recover=signed-integer-overflow -fsanitize=address")
endif(USE_SYCL)

# For better performance on A100, the option
# -DCUDA_ARCHITECTURES="70-real;80-real" can be added to cmake command
if (NOT USE_SYCL)
  if(DEFINED CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")
  else()
    set(CMAKE_CUDA_ARCHITECTURES "70-real;80-real;90-real")
  endif()
  message("CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
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

if(USE_SYCL)
  option(BUILD_CUTLASS "Using cutlass-SYCL" OFF)

  if(BUILD_CUTLASS)
    include(ExternalProject)

    # Detect if the compiler is IntelLLVM
    if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
      message(STATUS "IntelLLVM compiler detected: configuring CUTLASS for SYCL on Intel GPUs")

      # Set environment variables specific to IntelLLVM
      set(CUTLASS_ENV_VARS
        "SYCL_PROGRAM_COMPILE_OPTIONS=-ze-opt-large-register-file"
        "IGC_VISAOptions=-perfmodel"
        "IGC_VectorAliasBBThreshold=10000"
        "IGC_ExtraOCLOptions=-cl-intel-256-GRF-per-thread"
      )

      # CMake flags specific to IntelLLVM + SYCL
      set(CUTLASS_CMAKE_ARGS
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCUTLASS_ENABLE_SYCL=ON
        -DDPCPP_SYCL_TARGET=intel_gpu_pvc
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
        -DCUTLASS_ENABLE_EXAMPLES=OFF
        -DCUTLASS_ENABLE_BENCHMARKS=OFF
        -DCUTLASS_ENABLE_TESTS=OFF
        -DCMAKE_CXX_FLAGS=-ftemplate-backtrace-limit=0\ -fdiagnostics-color=always
      )
    else()
      message(WARNING "CUTLASS SYCL setup is only fully supported with IntelLLVM compiler (icx/icpx). Skipping custom flags.")
      set(CUTLASS_ENV_VARS "")
      set(CUTLASS_CMAKE_ARGS "")
    endif(CMAKE_CXX_COMPILER_ID)

    # Define the external CUTLASS SYCL project
    ExternalProject_Add(cutlass
      GIT_REPOSITORY https://github.com/codeplaysoftware/cutlass-sycl.git
      GIT_TAG        sycl-develop
      PREFIX         ${CMAKE_BINARY_DIR}/_deps/cutlass
      CMAKE_ARGS     ${CUTLASS_CMAKE_ARGS}
      CONFIGURE_ENVIRONMENT ${CUTLASS_ENV_VARS}
      BUILD_ENVIRONMENT     ${CUTLASS_ENV_VARS}
      INSTALL_ENVIRONMENT   ${CUTLASS_ENV_VARS}
      TEST_ENVIRONMENT      ${CUTLASS_ENV_VARS}
    )

    # Expose the source dir of CUTLASS
    ExternalProject_Get_Property(cutlass SOURCE_DIR)
    set(cutlass_SOURCE_DIR ${SOURCE_DIR})
  endif(BUILD_CUTLASS)
else()
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
endif()

add_subdirectory(gint)
add_subdirectory(gvhf)
add_subdirectory(gdft)
option(BUILD_CUPY_HELPER "Using cupy_helper" ON)
if(BUILD_CUPY_HELPER)
  if (USE_SYCL)
    add_subdirectory(onemkl_helper)
  endif()
  add_subdirectory(cupy_helper)
endif()

option(BUILD_SOLVENT "Using SMD solvent" ON)
if(BUILD_SOLVENT)
  add_subdirectory(solvent)
endif()

add_subdirectory(gvhf-rys)
add_subdirectory(gvhf-md)
add_subdirectory(pbc)
add_subdirectory(multigrid)
add_subdirectory(ecp)

option(BUILD_LIBXC "Using libxc for DFT" ON)
if(BUILD_LIBXC)
  if (USE_SYCL)

    find_package( ExchCXX REQUIRED CONFIG )
    message(STATUS "ExchCXX_DIR: ${ExchCXX_DIR}")
    message(STATUS "ExchCXX_FOUND: ${ExchCXX_FOUND}")

    if( NOT ${ExchCXX_FOUND} )
      message(STATUS "ExchCXX not found; fetching...")

      include(ExternalProject)
      ExternalProject_Add(ExchCXX
        GIT_REPOSITORY https://github.com/abagusetty/ExchCXX.git
        GIT_TAG master
        PREFIX ${PROJECT_BINARY_DIR}/deps
        INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=RelWithDebInfo
        -DBUILD_SHARED_LIBS=ON -DEXCHCXX_ENABLE_SYCL=ON -DEXCHCXX_SYCL_TARGET=intel_gpu_pvc
        -DCMAKE_BUILD_TYPE=RelWithDebInfo -DEXCHCXX_ENABLE_TESTS=OFF -DEXCHCXX_ENABLE_LIBXC=OFF
        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        -DCMAKE_INSTALL_LIBDIR:PATH=lib
        -DCMAKE_C_CREATE_SHARED_LIBRARY=${C_LINK_TEMPLATE}
        CMAKE_CACHE_ARGS
      )
    endif()

    # ExternalProject_Add(libxc
    #   GIT_REPOSITORY https://github.com/abagusetty/libxc.git
    #   GIT_TAG sycl
    #   PREFIX ${PROJECT_BINARY_DIR}/deps
    #   INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    #   CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    #   -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF -DENABLE_SYCL=ON
    #   -DENABLE_FORTRAN=OFF -DDISABLE_KXC=OFF -DDISABLE_LXC=ON -DDISABLE_FHC=ON
    #   -DCMAKE_CXX_COMPILER=icpx
    #   "-DCMAKE_CXX_FLAGS=-march=sapphirerapids -mtune=sapphirerapids -mlong-double-64 -fsycl -Wsycl-strict -Wexpected-file-type -fsycl-device-code-split=off -fsycl-targets=intel_gpu_pvc -sycl-std=2020 -fsycl-enable-function-pointers -fp-model=precise -Wexpected-file-type"
    #   -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
    #   -DCMAKE_INSTALL_LIBDIR:PATH=lib
    #   -DCMAKE_C_CREATE_SHARED_LIBRARY=${C_LINK_TEMPLATE}
    #   -DENABLE_XHOST:STRING=${BUILD_MARCH_NATIVE}
    #   CMAKE_CACHE_ARGS
    # )
  else (USE_SYCL)
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
  endif(USE_SYCL)
endif(BUILD_LIBXC)

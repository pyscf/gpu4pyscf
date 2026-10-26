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

cmake_minimum_required (VERSION 3.21 FATAL_ERROR) # Required by Libxc 7.1.2

# Backend selection — mutually exclusive (like ExchCXX)
option(USE_SYCL "Using SYCL backend" OFF)
option(USE_CUDA "Using CUDA backend" ON)

if(USE_SYCL AND USE_CUDA)
  message(FATAL_ERROR "SYCL and CUDA backends are mutually exclusive; choose one")
endif()
if(NOT USE_SYCL AND NOT USE_CUDA)
  message(FATAL_ERROR "Must select at least one backend: USE_SYCL=ON or USE_CUDA=ON")
endif()

project (gpu4pyscf C CXX Fortran)

set(CMAKE_C_STANDARD "99")
set(CMAKE_CXX_STANDARD 17)
if (USE_SYCL)
  # NTTP templates (template <auto Kernel>) in the launch launcher require
  # C++20; SYCL TUs are compiled with -x c++, so this applies only there.
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (USE_SYCL)
  set(DPCTL_CMAKE_MODULES_PATH "${PROJECT_SOURCE_DIR}/cmake")
  list(APPEND CMAKE_MODULE_PATH "${DPCTL_CMAKE_MODULES_PATH}")

  add_compile_definitions(USE_SYCL=1)

  # Production builds AOT-compile device code for Intel PVC (spir64_gen +
  # -device pvc). That cannot run anywhere without a PVC, so CPU-only CI
  # (GitHub-hosted runners driving the OpenCL CPU device) passes
  # -DGPU4PYSCF_SYCL_AOT=OFF to get a generic JIT (spir64) build instead.
  # Default ON keeps production PVC builds byte-identical.
  option(GPU4PYSCF_SYCL_AOT "AOT-compile SYCL device code for Intel PVC" ON)

  # Honor a user-supplied CMAKE_BUILD_TYPE (e.g. passed through
  # CMAKE_CONFIGURE_ARGS="-DCMAKE_BUILD_TYPE=RelWithDebInfo"). When nothing is
  # supplied, default to Release.
  #
  # With AOT (-fsycl-targets=spir64_gen) a -g injected by RelWithDebInfo/Debug
  # into CMAKE_CXX_FLAGS_<CFG> is forwarded into the device path and reaches
  # `ocloc -options "-g"`, which compiles device code in debug mode (disables
  # optimization). To keep the device backend optimized while still giving host
  # frames file:line for gdb-oneapi, we downgrade the C++ -g to
  # -gline-tables-only below. Host-side full -g is re-added further down for the
  # host-only (C/Fortran) translation units, which never carry SYCL device code.
  if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  endif()
  message(STATUS "SYCL build: CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

  # Downgrade any -g the selected build type injects into the C++ per-config
  # flags to host-only line tables. This keeps ocloc device compilation
  # optimized (no -options "-g") while preserving host file:line info. C and
  # Fortran host TUs keep full -g via the generator expressions below.
  foreach(_cfg RELWITHDEBINFO DEBUG)
    if (CMAKE_CXX_FLAGS_${_cfg})
      string(REPLACE "-g" "-gline-tables-only"
             CMAKE_CXX_FLAGS_${_cfg} "${CMAKE_CXX_FLAGS_${_cfg}}")
    endif()
  endforeach()

  add_compile_options(
    -fsycl
    -Wno-ignored-reference-qualifiers
    -Wno-register
    -fp-model=precise
    -fno-system-debug
    -fsycl-default-sub-group-size=32
  )
  if (GPU4PYSCF_SYCL_AOT)
    add_compile_options(
      -fsycl-targets=spir64_gen
      "SHELL:-Xsycl-target-backend=spir64_gen \"-device pvc\""
    )
  endif()

  # Host-side debug info WITHOUT propagating -g to the SYCL device backend.
  # Restrict -g to C and Fortran translation units (pure host code). SYCL
  # device-bearing C++ TUs are intentionally left without -g so `ocloc` does
  # not receive -options "-g" and the device build stays optimized.
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:C>:-g>"
    "$<$<COMPILE_LANGUAGE:Fortran>:-g>"
  )

  # Device code is split in sycl-post-link at LINK time, so the split mode
  # must be a link option (as a compile option it is ignored). per_kernel
  # gives llvm-foreach one ocloc job per kernel; without it a single-TU
  # library (ecp: nr_ecp_driver.cu) becomes one module compiled serially.
  add_link_options(
    -fsycl
    -flink-huge-device-code
    -fsycl-device-code-split=per_kernel
    -fsycl-max-parallel-link-jobs=64
  )
  if (GPU4PYSCF_SYCL_AOT)
    add_link_options(
      -ftarget-register-alloc-mode=pvc:auto
      -fsycl-targets=spir64_gen
      "SHELL:-Xsycl-target-backend=spir64_gen \"-device pvc\""
    )
    # ExchCXX validates this against a GPU-only allowlist; an empty value
    # disables its AOT pass entirely (generic spir64 JIT), which is what the
    # CPU-only CI needs.
    set(GPU4PYSCF_EXCHCXX_SYCL_TARGET intel_gpu_pvc)
  else()
    set(GPU4PYSCF_EXCHCXX_SYCL_TARGET "")
  endif()

  # This needs to be built first than anything else
  add_subdirectory(gsycl)
  # Global fallback so subdirectories that haven't yet gained
  # target_link_libraries(... sycl_compat) still find
  # <cuda_runtime.h> and <cuda.h> via gsycl/ shims.
  # Remove once all subdirectory CMakeLists.txt are updated.
  include_directories(${PROJECT_SOURCE_DIR}/gsycl)      
endif()

if (NOT USE_SYCL)
  # Set architectures before enabling CUDA, which otherwise initializes its own default.
  # For better performance on A100, the option
  # -DCUDA_ARCHITECTURES="70-real;80-real" can be added to cmake command
  if(DEFINED CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")
  elseif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70-real;80-real;90-real")
  endif()
  enable_language(CUDA)
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

option(ENABLE_MULTIGRID_V2 "Enable multigrid-v2 kernels" ON)
option(ENABLE_FP32_MULTIGRID "Enable experimental FP32 multigrid-v2 collocation kernels" OFF)

if(USE_SYCL)
  add_subdirectory(onemkl_helper)

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

# ExchCXX / libxc configuration — mutually exclusive.
# Defaults are backend-aware so a SYCL configure never starts with
# BUILD_LIBXC=ON (libxc is CUDA-only; SYCL uses ExchCXX).
if(USE_SYCL)
  option(BUILD_EXCHCXX "Build ExchCXX for SYCL backend (XC functionals)" ON)
  option(BUILD_LIBXC "Build libxc for CUDA backend (XC functionals)" OFF)
else()
  option(BUILD_EXCHCXX "Build ExchCXX for SYCL backend (XC functionals)" OFF)
  option(BUILD_LIBXC "Build libxc for CUDA backend (XC functionals)" ON)
endif()

# Enforce correct combination based on backend. FORCE is the safety net for
# stale caches and explicit -D flags that request an invalid combination
# (e.g. a cached or user-passed BUILD_LIBXC=ON with USE_SYCL=ON).
if(USE_SYCL)
  set(BUILD_EXCHCXX ON CACHE BOOL "Build ExchCXX for SYCL" FORCE)
  set(BUILD_LIBXC OFF CACHE BOOL "Using libxc for DFT" FORCE)
  message(STATUS "SYCL backend: BUILD_EXCHCXX=ON, BUILD_LIBXC=OFF")
else()
  set(BUILD_EXCHCXX OFF CACHE BOOL "Build ExchCXX for SYCL" FORCE)
  # BUILD_LIBXC keeps user/default value (ON)
endif()

# ExchCXX setup (SYCL only) — must run before gdft which links ExchCXX::ExchCXX
if(BUILD_EXCHCXX)
  include(ExternalProject)

  set(_exchcxx_lib "${PROJECT_SOURCE_DIR}/deps/lib/libexchcxx.so")
  set(_exchcxx_inc "${PROJECT_SOURCE_DIR}/deps/include")

  # Always define the IMPORTED target
  file(MAKE_DIRECTORY "${_exchcxx_inc}")
  file(MAKE_DIRECTORY "${PROJECT_SOURCE_DIR}/deps/lib")

  add_library(ExchCXX::ExchCXX SHARED IMPORTED GLOBAL)
  set_target_properties(ExchCXX::ExchCXX PROPERTIES
    IMPORTED_LOCATION             "${_exchcxx_lib}"
    # The ExchCXX sub-build overrides CMAKE_CXX_CREATE_SHARED_LIBRARY, which drops
    # -soname, so libexchcxx.so has no SONAME. Without this, libgdft.so/libxc.so
    # record the absolute path of libexchcxx.so as DT_NEEDED and break as soon as
    # the build tree moves or is deleted (e.g. CI publishing the built tree).
    IMPORTED_NO_SONAME            TRUE
    INTERFACE_INCLUDE_DIRECTORIES "${_exchcxx_inc}"
    INTERFACE_COMPILE_DEFINITIONS "EXCHCXX_HAS_CONFIG_H=1"
  )

  # Conditionally build or skip ExchCXX
  if(NOT EXISTS "${_exchcxx_lib}")
    message(STATUS "ExchCXX: not found -- building from source (~35 min)")
    ExternalProject_Add(ExchCXX
      GIT_REPOSITORY    https://github.com/abagusetty/ExchCXX.git
      GIT_TAG           1274182f28624fd173d758051605908827ebe9a4
      UPDATE_DISCONNECTED ON
      PREFIX            ${PROJECT_BINARY_DIR}/deps
      INSTALL_DIR       ${PROJECT_SOURCE_DIR}/deps
      CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=Release
      -DBUILD_SHARED_LIBS=ON
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DEXCHCXX_ENABLE_SYCL=ON
      -DEXCHCXX_SYCL_TARGET=${GPU4PYSCF_EXCHCXX_SYCL_TARGET}
      -DEXCHCXX_ENABLE_TESTS=OFF
      -DEXCHCXX_ENABLE_LIBXC=OFF
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
      -DCMAKE_INSTALL_LIBDIR:PATH=lib
      -DCMAKE_C_CREATE_SHARED_LIBRARY=${C_LINK_TEMPLATE}
      -DCMAKE_CXX_CREATE_SHARED_LIBRARY=${CXX_LINK_TEMPLATE}
      BUILD_BYPRODUCTS "${_exchcxx_lib}"
    )
  else()
    message(STATUS "ExchCXX: found ${_exchcxx_lib} -- skipping rebuild")
    add_custom_target(ExchCXX)
  endif()

  add_dependencies(ExchCXX::ExchCXX ExchCXX)
endif()

# libxc setup (CUDA only) — runs at end like upstream
if(BUILD_LIBXC)
  include(ExternalProject)
  ExternalProject_Add(libxc
    URL https://gitlab.com/libxc/libxc/-/archive/7.1.2/libxc-7.1.2.tar.gz
    PATCH_COMMAND patch -p1 --batch --forward -i ${PROJECT_SOURCE_DIR}/libxc_prune.patch
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    CMAKE_ARGS -DBUILD_SHARED_LIBS=ON -DENABLE_CUDA=ON
            -DBUILD_TESTING=OFF -DENABLE_FORTRAN=OFF
            -DMAXORDER=3 -DDISABLE_FHC=ON
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
            -DCMAKE_INSTALL_LIBDIR:PATH=lib
            -DCMAKE_C_CREATE_SHARED_LIBRARY=${C_LINK_TEMPLATE}
            -DENABLE_XHOST:STRING=${BUILD_MARCH_NATIVE}
    CMAKE_CACHE_ARGS -DCMAKE_CUDA_ARCHITECTURES:STRING=${CMAKE_CUDA_ARCHITECTURES}
  )
endif()

add_subdirectory(gint)
add_subdirectory(gvhf)
add_subdirectory(gdft)
add_subdirectory(cupy_helper)
add_subdirectory(sem)

option(BUILD_SOLVENT "Using SMD solvent" ON)
if(BUILD_SOLVENT)
  add_subdirectory(solvent)
endif()

add_subdirectory(gvhf-rys)
add_subdirectory(gvhf-md)
add_subdirectory(pbc)
add_subdirectory(multigrid)
add_subdirectory(ecp)

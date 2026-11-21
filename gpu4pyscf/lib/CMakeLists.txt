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

cmake_minimum_required (VERSION 3.21 FATAL_ERROR) # Required by Libxc 7.1.2 and the first-class HIP language
project (gpu4pyscf C CXX)

set(GPU4PYSCF_BACKEND cuda CACHE STRING "GPU backend: cuda, hip, or sycl")
set_property(CACHE GPU4PYSCF_BACKEND PROPERTY STRINGS cuda hip sycl)
string(TOLOWER "${GPU4PYSCF_BACKEND}" GPU4PYSCF_BACKEND)
set(USE_HIP OFF)
set(USE_SYCL OFF)
if(GPU4PYSCF_BACKEND STREQUAL "hip")
  set(USE_HIP ON)
elseif(GPU4PYSCF_BACKEND STREQUAL "sycl")
  set(USE_SYCL ON)
elseif(NOT GPU4PYSCF_BACKEND STREQUAL "cuda")
  message(FATAL_ERROR "GPU4PYSCF_BACKEND must be cuda, hip, or sycl")
endif()

if(USE_HIP)
  # Set explicit targets before compiler detection so builds do not need a GPU.
  # HIP_ARCHITECTURES takes precedence over CMake's native architecture setting.
  set(_hip_cdna_archs "gfx908;gfx90a;gfx942;gfx950")
  set(_hip_rdna_archs "gfx10-3-generic;gfx11-generic;gfx12-generic")
  string(TOLOWER "${HIP_ARCHITECTURES}" _hip_target)
  if(_hip_target STREQUAL "all")
    set(CMAKE_HIP_ARCHITECTURES "${_hip_cdna_archs};${_hip_rdna_archs}")
  elseif(_hip_target STREQUAL "cdna")
    set(CMAKE_HIP_ARCHITECTURES "${_hip_cdna_archs}")
  elseif(_hip_target STREQUAL "rdna")
    set(CMAKE_HIP_ARCHITECTURES "${_hip_rdna_archs}")
  elseif(_hip_target STREQUAL "native" OR _hip_target STREQUAL "auto")
    # An explicit request to detect GPUs overrides any cached architecture list.
    set(CMAKE_HIP_ARCHITECTURES "")
  elseif(HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "${HIP_ARCHITECTURES}")
  endif()

  # Preserve an explicit CMake architecture list; otherwise detect installed GPUs.
  if(NOT CMAKE_HIP_ARCHITECTURES)
    find_program(ROCM_AGENT_ENUMERATOR NAMES rocm_agent_enumerator
                 HINTS ENV ROCM_PATH ENV HIP_PATH ENV ROCM_HOME PATH_SUFFIXES bin)
    if(ROCM_AGENT_ENUMERATOR)
      execute_process(COMMAND ${ROCM_AGENT_ENUMERATOR}
                      OUTPUT_VARIABLE _rocm_agents
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
      string(REPLACE "\n" ";" _rocm_agents "${_rocm_agents}")
      set(_hip_archs "")
      foreach(_agent ${_rocm_agents})
        # skip the host/CPU agent (gfx000) and any blank lines
        if(_agent MATCHES "^gfx" AND NOT _agent STREQUAL "gfx000")
          list(APPEND _hip_archs "${_agent}")
        endif()
      endforeach()
      list(REMOVE_DUPLICATES _hip_archs)
      set(CMAKE_HIP_ARCHITECTURES "${_hip_archs}")
    endif()
  endif()
  if(NOT CMAKE_HIP_ARCHITECTURES)
    message(FATAL_ERROR
      "HIP backend selected but no AMD GPU architecture could be determined. "
      "Pass -DHIP_ARCHITECTURES=gfxXXXX (e.g. gfx90a, gfx942, gfx1100, gfx1151) "
      "or set CMAKE_HIP_ARCHITECTURES.")
  endif()
  enable_language(HIP)
endif()

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

  # Generic device code is compiled for the selected device at runtime.
  # PVC instead compiles ahead of time for Intel Data Center GPU Max.
  set(SYCL_ARCHITECTURES generic CACHE STRING "SYCL target: generic or pvc")
  set_property(CACHE SYCL_ARCHITECTURES PROPERTY STRINGS generic pvc)
  string(TOLOWER "${SYCL_ARCHITECTURES}" SYCL_ARCHITECTURES)
  if(NOT SYCL_ARCHITECTURES MATCHES "^(generic|pvc)$")
    message(FATAL_ERROR "SYCL_ARCHITECTURES must be generic or pvc")
  endif()

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
  if (SYCL_ARCHITECTURES STREQUAL "pvc")
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
  if (SYCL_ARCHITECTURES STREQUAL "pvc")
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

  # Build the libgsycl runtime and define the SYCL header target only for SYCL builds.
  add_subdirectory(gsycl)
  # GPU targets link the sycl_compat INTERFACE target to add compat/sycl/ to
  # their include path. This resolves <cuComplex.h> and <cub/cub.cuh> to the
  # SYCL shims while CUDA builds use the toolkit headers.
elseif(USE_HIP)
  message("HIP_ARCHITECTURES: ${CMAKE_HIP_ARCHITECTURES}")
else()
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
include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/compat)
include_directories(${PROJECT_SOURCE_DIR}/deps/include)
link_directories(${PROJECT_SOURCE_DIR}/deps/lib)

if(USE_HIP)
  # Place the CUDA->HIP compatibility stubs (cuda_runtime.h / cuda.h) ahead of
  # the ROCm headers, and force-include the shim so the CUDA-authored .cu
  # kernels build unmodified with hipcc.
  include_directories(BEFORE ${PROJECT_SOURCE_DIR}/compat/hip)
  # -Wno-register: the CUDA sources use the C++17-removed `register` keyword.
  # gpu_runtime.h: CUDA->HIP shim (force-included). warp_size.h: per-arch WARP_SIZE
  # + the host effective-warp-size helper (force-included after the shim, so the
  # HIP runtime it uses is already available).
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Wno-register -include ${PROJECT_SOURCE_DIR}/compat/hip/gpu_runtime.h -include ${PROJECT_SOURCE_DIR}/compat/warp_size.h")

  # Wavefront width is derived per device sub-compilation from the target
  # __gfxNNN__ macro in the force-included warp_size.h (64 on GCN/CDNA, 32 on
  # RDNA+), so a single fat binary is native on every listed arch with no global
  # flag -- host-side sizing uses gpu4pyscf_effective_warp_size() (runtime
  # warpSize), so the host pass' fallback of 32 is harmless. We only classify the
  # arches here to decide the gint nrys>=7 cap: that kernel's stack frame exceeds
  # the amdgpu limit under wave64, so it is dropped whenever any wave64 arch is in
  # the build (a mixed binary also caps it for its wave32 arch(es) -- a niche
  # high-angular-momentum path).
  set(_wave64_archs "")
  foreach(_a IN LISTS CMAKE_HIP_ARCHITECTURES)
    if(_a MATCHES "^gfx[89]")   # gfx8xx / gfx9xx = GCN/CDNA = wave64
      list(APPEND _wave64_archs "${_a}")
    endif()
  endforeach()

  # Manual override: force wave64 even on RDNA to validate the wave64 code path on
  # an RDNA dev GPU (e.g. gfx1151). -mwavefrontsize64 switches RDNA into wave64
  # mode; -DWARP_SIZE=64 overrides the per-arch derivation for every sub-compile.
  option(USE_WAVE64 "Force wave64 wavefronts (RDNA dev-validation of wave64 kernels)" OFF)
  if(USE_WAVE64)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -mwavefrontsize64 -DWARP_SIZE=64 -DGPU4PYSCF_WAVE64=1")
    message("USE_WAVE64: forcing wave64 (-mwavefrontsize64 -DWARP_SIZE=64)")
  elseif(_wave64_archs)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -DGPU4PYSCF_WAVE64=1")
    message("Wavefront: per-arch native (wave64 arch(es): ${_wave64_archs}; gint nrys>=7 capped)")
  else()
    message("Wavefront: per-arch native (all RDNA/wave32: ${CMAKE_HIP_ARCHITECTURES})")
  endif()
endif()

# Configure a GPU library target for the HIP and SYCL backend: compile the target's .cu
# sources as HIP and apply separable compilation + arch. On CUDA builds this is
# a no-op, so each sub-project keeps its existing CUDA_* target properties
# unchanged (those are simply inert when USE_HIP is ON).
function(gpu4pyscf_configure_gpu_target target)
  if(USE_SYCL)
    get_target_property(_gpu_srcs ${target} SOURCES)
    foreach(_src ${_gpu_srcs})
      # Compile C sources as C++ too: SYCL flags are incompatible with C99.
      if(_src MATCHES "\\.(c|cu)$")
        set_source_files_properties(${_src} PROPERTIES LANGUAGE CXX)
      endif()
    endforeach()
    # Tell the SYCL compiler to read the .cu files as C++ and skip the CUDA
    # headers and device libraries it would normally add for CUDA compilation.
    # Apply these flags only to C++ files: targets such as solvent also contain
    # Fortran files, whose compiler does not accept these options.
    target_compile_options(${target} PRIVATE
      "$<$<COMPILE_LANGUAGE:CXX>:-x;c++;-nocudainc;-nocudalib>")
    target_link_libraries(${target} PRIVATE sycl_compat gsycl)
  elseif(USE_HIP)
    get_target_property(_gpu_srcs ${target} SOURCES)
    foreach(_src ${_gpu_srcs})
      if(_src MATCHES "\\.cu$")
        set_source_files_properties(${_src} PROPERTIES LANGUAGE HIP)
      endif()
    endforeach()
    # Relocatable device code so __device__/__constant__ globals declared in one
    # translation unit and used in another link across the library (the CUDA
    # build relies on CUDA_SEPARABLE_COMPILATION for the same reason). Set the
    # flag explicitly at compile and link time -- HIP_SEPARABLE_COMPILATION alone
    # does not reliably add -fgpu-rdc at compile time.
    set_target_properties(${target} PROPERTIES
      HIP_SEPARABLE_COMPILATION ON
      HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>)
    target_link_options(${target} PRIVATE -fgpu-rdc --hip-link)
  endif()
endfunction()

#list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

set(C_LINK_TEMPLATE "<CMAKE_C_COMPILER> <CMAKE_SHARED_LIBRARY_C_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
set(CXX_LINK_TEMPLATE "<CMAKE_CXX_COMPILER> <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")

option(ENABLE_MULTIGRID_V2 "Enable multigrid-v2 kernels" OFF)
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
  if(USE_HIP)
    option(BUILD_CUTLASS "Using cutlass" OFF)
  else()
    option(BUILD_CUTLASS "Using cutlass" ON)
  endif()
  if(BUILD_CUTLASS)
    include(ExternalProject)
    ExternalProject_Add(cutlass
      GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
      GIT_TAG        v3.6.0
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
  if(USE_HIP)
    option(BUILD_LIBXC "Using libxc for DFT" OFF)
  else()
    option(BUILD_LIBXC "Using libxc for DFT" ON)
  endif()
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
  enable_language(Fortran)
  add_subdirectory(solvent)
endif()

add_subdirectory(gvhf-rys)
add_subdirectory(gvhf-md)
add_subdirectory(pbc)
add_subdirectory(multigrid)
add_subdirectory(ecp)

# Python imports this value before initializing backend-specific modules.
file(GENERATE OUTPUT "${PROJECT_SOURCE_DIR}/../_build_config.py"
     CONTENT "# Generated by CMake.\nBACKEND = '${GPU4PYSCF_BACKEND}'\n")

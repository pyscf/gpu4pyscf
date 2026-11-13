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

# ---------------------------------------------------------------------------
# GPU backend selection: NVIDIA CUDA (default) or AMD HIP/ROCm (-DUSE_HIP=ON).
# The two paths are mutually exclusive and the CUDA path is left unchanged
# (CUDA is enabled below, after its architectures are set).
# Fortran is only needed by the optional solvent module and is enabled on
# demand (see below), so neither backend requires a Fortran compiler unless
# solvent is actually built.
# ---------------------------------------------------------------------------
option(USE_HIP "Build GPU kernels with AMD HIP/ROCm instead of NVIDIA CUDA" OFF)
if(USE_HIP)
  # Set explicit targets before compiler detection so builds do not need a GPU.
  if(DEFINED GPU_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "${GPU_ARCHITECTURES}")
  endif()
  enable_language(HIP)
endif()

set(CMAKE_C_STANDARD "99")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GPU architecture selection.
if(USE_HIP)
  # AMD GPU targets. Honour the explicit override set above or a pre-set
  # CMAKE_HIP_ARCHITECTURES; otherwise auto-detect the installed AMD GPU(s)
  # with rocm_agent_enumerator. No specific gfx arch is hard-coded.
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
      "USE_HIP is ON but no AMD GPU architecture could be determined. "
      "Pass -DGPU_ARCHITECTURES=gfxXXXX (e.g. gfx90a, gfx942, gfx1100, gfx1151) "
      "or set CMAKE_HIP_ARCHITECTURES.")
  endif()
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
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/deps/include)
link_directories(${PROJECT_SOURCE_DIR}/deps/lib)

if(USE_HIP)
  # Place the CUDA->HIP compatibility stubs (cuda_runtime.h / cuda.h) ahead of
  # the ROCm headers, and force-include the shim so the CUDA-authored .cu
  # kernels build unmodified with hipcc.
  include_directories(BEFORE ${PROJECT_SOURCE_DIR}/hip_compat)
  # -Wno-register: the CUDA sources use the C++17-removed `register` keyword.
  # gpu_runtime.h: CUDA->HIP shim (force-included). warp_size.h: per-arch WARP_SIZE
  # + the host effective-warp-size helper (force-included after the shim, so the
  # HIP runtime it uses is already available).
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Wno-register -include ${PROJECT_SOURCE_DIR}/hip_compat/gpu_runtime.h -include ${PROJECT_SOURCE_DIR}/warp_size.h")

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

# Configure a GPU library target for the HIP backend: compile the target's .cu
# sources as HIP and apply separable compilation + arch. On CUDA builds this is
# a no-op, so each sub-project keeps its existing CUDA_* target properties
# unchanged (those are simply inert when USE_HIP is ON).
function(gpu4pyscf_configure_gpu_target target)
  if(USE_HIP)
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

option(ENABLE_MULTIGRID_V2 "Enable multigrid-v2 kernels" ON)
option(ENABLE_FP32_MULTIGRID "Enable experimental FP32 multigrid-v2 collocation kernels" OFF)

# CUTLASS is NVIDIA-only; default off on HIP (a CuPy fallback replaces the
# grouped GEMM/dot kernels there).
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

add_subdirectory(gint)
add_subdirectory(gvhf)
add_subdirectory(gdft)
add_subdirectory(cupy_helper)
add_subdirectory(sem)
add_subdirectory(gvhf-rys)
add_subdirectory(gvhf-md)
add_subdirectory(pbc)
add_subdirectory(multigrid)
add_subdirectory(ecp)

option(BUILD_SOLVENT "Using SMD solvent" ON)
if(BUILD_SOLVENT)
  enable_language(Fortran)   # solvent has Fortran sources (mnsol)
  add_subdirectory(solvent)
endif()

# The bundled libxc build is CUDA-only (ENABLE_CUDA). Default off on HIP; a
# HIP-enabled libxc is wired up in a later phase.
if(USE_HIP)
  option(BUILD_LIBXC "Using libxc for DFT" OFF)
else()
  option(BUILD_LIBXC "Using libxc for DFT" ON)
endif()
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

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

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(GPU_SRCS
  nr_eval_gto.cu
  contract_rho.cu
  gen_grids.cu
  nr_numint_sparse.cu
  vv10.cu
)

if(USE_SYCL)
  # ── Step 1: Compile exchcxx.cpp ONCE as an OBJECT library ─────────────
  # Isolated from GPU_SRCS so changes to grid/numint kernels never
  # retrigger this expensive AoT SPIR-V compilation (~35 min).
  # Only rebuilt when exchcxx.cpp itself changes.
  add_library(exchcxx_shim OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/exchcxx.cpp
  )
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/exchcxx.cpp PROPERTIES LANGUAGE CXX
  )
  set_target_properties(exchcxx_shim PROPERTIES
    POSITION_INDEPENDENT_CODE ON
  )
  target_compile_options(exchcxx_shim PRIVATE
    -x c++ -nocudainc -nocudalib
  )
  target_include_directories(exchcxx_shim PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}   # exchcxx.h, gint/sycl_device.hpp
    ${PROJECT_SOURCE_DIR}         # project-wide includes
  )
  # Provides exchcxx/exchcxx.hpp include path + EXCHCXX_HAS_CONFIG_H=1
  target_link_libraries(exchcxx_shim PRIVATE ExchCXX::ExchCXX)
  add_dependencies(exchcxx_shim ExchCXX)

  # ── Step 2: libgdft.so ─────────────────────────────────────────────────
  # GPU grid/numint kernels + exchcxx.o via TARGET_OBJECTS (no recompile).
  # Python libgdft handle: GDFT_xc_lda/gga/mgga entry points live here.
  # Changing any GPU_SRCS file only recompiles that file + relinks gdft.
  file(GLOB CUH_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh")
  list(APPEND CUH_HEADERS exchcxx.h)        # .h suffix not caught by *.cuh glob
  add_library(gdft SHARED
    ${GPU_SRCS}
    ${CUH_HEADERS}
    $<TARGET_OBJECTS:exchcxx_shim>           # reuse compiled object, never recompiles
  )
  set_source_files_properties(${GPU_SRCS} PROPERTIES LANGUAGE CXX)
  target_compile_options(gdft PRIVATE -x c++ -nocudainc -nocudalib)
  # Required: exchcxx.o inside libgdft.so calls ExchCXX device kernel symbols
  # that must be resolved against libexchcxx.so at runtime
  target_link_libraries(gdft PRIVATE ExchCXX::ExchCXX)

  # ── Step 3: libxc.so — the libxc ABI shim Python loads as _libxc ──────
  # Python _libxc handle: xc_func_alloc/init/end/free +
  #                       xc_version_string + xc_number_of_functionals etc.
  # Same exchcxx.o reused via TARGET_OBJECTS — zero recompilation cost.
  # Replaces the old create_symlink approach with a real DSO target.
  add_library(xc SHARED
    $<TARGET_OBJECTS:exchcxx_shim>           # reuse same compiled object
  )
  # Required: exchcxx.o inside libxc.so calls ExchCXX device kernel symbols
  target_link_libraries(xc PRIVATE ExchCXX::ExchCXX)
  set_target_properties(xc PROPERTIES
    OUTPUT_NAME              xc
    # Python searches gpu4pyscf/lib/deps/lib/libxc.so
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/deps/lib"
    # libexchcxx.so is in the same deps/lib/ directory
    BUILD_RPATH              "$ORIGIN"
    INSTALL_RPATH            "$ORIGIN"
  )

else()
  list(APPEND GPU_SRCS libxc.cu)
  add_library(gdft SHARED ${GPU_SRCS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
endif()

set_target_properties(gdft PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)

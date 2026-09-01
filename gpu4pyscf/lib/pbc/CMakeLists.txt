set(GPU_SRCS
  pbc_driver.cu
  ft_ao.cu ft_ao_ip1.cu contract_ft_ao.cu
  fill_int3c2e.cu contract_int3c2e.cu ejk_int3c2e_ip1.cu
  fill_int2c2e.cu fill_int2c2e_ip1.cu
  contract_int3c2e_pvp.cu
  overlap.cu supmol_sr_estimator.cu
  rys_roots_dat.cu
  rys_contract_k.cu rys_contract_jk_ip1.cu
  rys_contract_j.cu
  unrolled_rys_k.cu
  nr_eval_gto.cu
  fill_triu.cu
  sorting.c
)

add_library(pbc SHARED ${GPU_SRCS})

set_target_properties(pbc PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

if (USE_SYCL)
  # icx rejects the project-wide -std=gnu99 (CMAKE_C_STANDARD 99) together
  # with -fsycl, so plain-C sources are compiled as C++ here, matching
  # gvhf-md/md_pairdata.c. sorting.c carries an extern "C" guard.
  set_source_files_properties(${GPU_SRCS} PROPERTIES LANGUAGE CXX)
  target_compile_options(pbc PRIVATE -x c++ -nocudainc -nocudalib)
  target_link_libraries(pbc PRIVATE sycl_compat gsycl)
else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
  set_target_properties(pbc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

target_link_libraries(pbc PRIVATE OpenMP::OpenMP_C)

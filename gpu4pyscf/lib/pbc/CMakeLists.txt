set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")# -maxrregcount=128")

add_library(pbc SHARED
  pbc_driver.cu ft_ao.cu unrolled_ft_ao.cu ft_ao_bdiv.cu
  ft_ao_ip1.cu
  fill_int3c2e.cu unrolled_int3c2e.cu fill_int2c2e.cu
  fill_int3c2e_v2.cu
  overlap.cu
  estimator.cu supmol_sr_estimator.cu
  rys_roots_dat.cu
  rys_contract_k.cu
  rys_contract_jk_ip1.cu
  nr_eval_gto.cu
  sorting.c
)

set_target_properties(pbc PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(pbc OpenMP::OpenMP_C)

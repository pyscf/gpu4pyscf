set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")# -maxrregcount=128")

add_library(pbc SHARED
  pbc_driver.cu ft_ao.cu unrolled_ft_ao.cu ft_ao_bdiv.cu
  fill_int3c2e.cu unrolled_int3c2e.cu fill_int2c2e.cu
  estimator.cu
  rys_roots_dat.cu
  sorting.c
)

set_target_properties(pbc PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(pbc OpenMP::OpenMP_C)

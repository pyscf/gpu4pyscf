set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")# -maxrregcount=128")

add_library(gint_rys SHARED
  gint_driver.cu fill_int3c2e.cu unrolled_int3c2e.cu
  fill_int3c2e_bdiv.cu unrolled_int3c2e_bdiv.cu
  rys_roots_dat.cu
)

set_target_properties(gint_rys PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  CUDA_SEPARABLE_COMPILATION ON)

#target_link_libraries(ft_ao OpenMP::OpenMP_C)

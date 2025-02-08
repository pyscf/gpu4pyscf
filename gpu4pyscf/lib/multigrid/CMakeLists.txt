set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")# -maxrregcount=128")

add_library(mgrid SHARED
  multigrid.cu mg_driver.cu
)

set_target_properties(mgrid PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  CUDA_SEPARABLE_COMPILATION ON)

#target_link_libraries(ft_ao OpenMP::OpenMP_C)

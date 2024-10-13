set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")# -maxrregcount=128")

add_library(gvhf_rys SHARED
  rys_contract_jk.cu rys_jk_driver.cu unrolled_os.cu unrolled_rys.cu
  nr_sr_estimator.c
  rys_contract_j.cu cart2xyz.c unrolled_rys_j.cu
  md_contract_j.cu md_j_driver.cu md_pairdata.c unrolled_md_j.cu
  count_tasks.cu
)

#option(BUILD_SHARED_LIBS "build shared libraries" 1)
#option(ENABLE_STATIC "Enforce static library build" 0)
#if(ENABLE_STATIC)
#  set(BUILD_SHARED_LIBS 0)
#endif()

set_target_properties(gvhf_rys PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(gvhf_rys OpenMP::OpenMP_C)

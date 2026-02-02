set(GPU_SRCS
  mg_driver.cu estimator.cu
  eval_rho.cu
  eval_mat_lda.cu eval_mat_gga.cu
  eval_tau.cu eval_mat_tau.cu
)

set(GPU_SRCS_v2
  multigrid_v2/drivers.cu
  multigrid_v2/screen.cu
  multigrid_v2/eval_xc.cu
  multigrid_v2/eval_xc_grad.cu
)

add_library(mgrid SHARED ${GPU_SRCS})
add_library(mgrid_v2 SHARED ${GPU_SRCS_v2})

if (USE_SYCL)
  file(GLOB CUH_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/multigrid_v2/*.cuh"
  )
  set_source_files_properties(${CUH_HEADERS} PROPERTIES LANGUAGE CXX)

  set_source_files_properties(${GPU_SRCS} PROPERTIES LANGUAGE CXX)
  set_source_files_properties(${GPU_SRCS_v2} PROPERTIES LANGUAGE CXX)  

  set_target_properties(mgrid PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
  target_compile_options(mgrid PRIVATE -x c++ -nocudainc -nocudalib)

  set_target_properties(mgrid_v2 PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
  target_compile_options(mgrid_v2 PRIVATE -x c++ -nocudainc -nocudalib)  
else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")# -maxrregcount=128")

  set_target_properties(mgrid PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    CUDA_SEPARABLE_COMPILATION ON)

  set_target_properties(mgrid_v2 PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    CUDA_SEPARABLE_COMPILATION ON)  
endif (USE_SYCL)

#target_link_libraries(ft_ao OpenMP::OpenMP_C)

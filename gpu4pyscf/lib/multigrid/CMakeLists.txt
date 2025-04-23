set(GPU_SRCS
    mg_driver.cu
    eval_rho.cu
    eval_mat_lda.cu eval_mat_gga.cu
    eval_tau.cu eval_mat_tau.cu
  )

if (USE_SYCL)
  file(GLOB CUH_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh")
  set_source_files_properties(${CUH_HEADERS} PROPERTIES LANGUAGE CXX)

  set_source_files_properties(${GPU_SRCS} PROPERTIES LANGUAGE CXX)
else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")# -maxrregcount=128")

  set_target_properties(mgrid PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    CUDA_SEPARABLE_COMPILATION ON)
endif (USE_SYCL)

add_library(mgrid SHARED ${GPU_SRCS})

#target_link_libraries(ft_ao OpenMP::OpenMP_C)


set(GPU_SRCS
  md_contract_j.cu unrolled_md_j.cu unrolled_md_j_4dm.cu
  md_indices.cu md_j_driver.cu md_pairdata.c 
  contract_int3c2e.cu
  pbc_md_contract_j.cu
)

add_library(gvhf_md SHARED ${GPU_SRCS})

#option(BUILD_SHARED_LIBS "build shared libraries" 1)
#option(ENABLE_STATIC "Enforce static library build" 0)
#if(ENABLE_STATIC)
#  set(BUILD_SHARED_LIBS 0)
#endif()

if (USE_SYCL)
  set_source_files_properties(${GPU_SRCS} PROPERTIES LANGUAGE CXX)
  set_target_properties(gvhf_md PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
  target_compile_options(gvhf_md PRIVATE -x c++ -nocudainc -nocudalib)
else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

  set_target_properties(gvhf_md PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    CUDA_SEPARABLE_COMPILATION ON)
endif()


target_link_libraries(gvhf_md OpenMP::OpenMP_C)

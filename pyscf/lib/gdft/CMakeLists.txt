set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")

add_library(gdft SHARED 
  nr_eval_gto.cu
  contract_rho.cu
  nr_numint_sparse.cu
)

set_target_properties(gdft PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

target_link_libraries(gdft cint)

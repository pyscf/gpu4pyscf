#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_30 -rdc=true")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60 --ptxas-options=-v")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_50 --ptxas-options=-v -maxrregcount=255")

add_library(gint SHARED
  g2e.c
  pair_data.c
  rys_roots.cu
  nr_fill_ao_ints.cu
  nr_fill_ao_ints_gradient.cu
)

#option(BUILD_SHARED_LIBS "build shared libraries" 1)
#option(ENABLE_STATIC "Enforce static library build" 0)
#if(ENABLE_STATIC)
#  set(BUILD_SHARED_LIBS 0)
#endif()

set_target_properties(gint PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
#  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  CUDA_SEPARABLE_COMPILATION ON)

#target_link_libraries(gint gint cint)
if(OPENMP_FOUND)
  target_link_libraries(gint
    PUBLIC OpenMP::OpenMP_C
    cint)
else()
  target_link_libraries(gint cint)
endif()


#find_package(CUDAToolkit REQUIRED)
find_package(NVHPC REQUIRED)

add_library(deepeeRT_dpc STATIC
  dpc.cpp
)
target_include_directories(deepeeRT_dpc
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}
)


target_link_libraries(deepeeRT_dpc PUBLIC
#  CUDA::cuda_driver
#  CUDA::cudart_static
)


macro(dpc_configure_source tgt)
  set_source_files_properties(${src} PROPERTIES LANGUAGE CPP)
  set_source_files_properties(${src} PROPERTIES COMPILE_OPTIONS
    "-fopenmp -foffload=nvptx-none=-march=sm_80"
  )
#  if (NOT WIN32)
#    set_source_files_properties(${src} PROPERTIES COMPILE_OPTIONS "-xc++")
#  endif()
endmacro()

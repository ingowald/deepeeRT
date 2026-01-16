# CC=nvc CXX=nvc++ cmake .. -DDPC_OMP=ON

#find_package(CUDAToolkit REQUIRED)
find_package(NVHPC COMPONENTS CUDA REQUIRED)

add_library(deepeeRT_dpc STATIC
  dpc.cpp
)
target_include_directories(deepeeRT_dpc
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}
)


target_link_libraries(deepeeRT_dpc PUBLIC
#  NVHPC::HOSTUTILS
  NVHPC::CUDA
#  CUDA::cuda_driver
#  CUDA::cudart_static
#  CUDA::cudart
)
target_link_options(deepeeRT_dpc PUBLIC "-cuda;-static-nvidia")

macro(dpc_configure_source src)
  set_source_files_properties(${src} PROPERTIES
#    LANGUAGE CUDA
    LANGUAGE CXX
  )
  set_source_files_properties(${src} PROPERTIES
    LANGUAGE CXX
    COMPILE_OPTIONS
#    "-fopenmp;-gpu=cc80;-stdpar" 
#    "-fopenmp;-gpu=cc80" 
    "-fopenmp;-gpu=ccnative;-mp=gpu" 
#    "-fopenmp;-gpu=cc80;-foffload=nvptx-none=-march=sm_80" 
#    "-fopenmp;-foffload=nvptx-none=-march=sm_80" 
  )
#  if (NOT WIN32)
#    set_source_files_properties(${src} PROPERTIES COMPILE_OPTIONS "-xc++")
#  endif()
endmacro()

macro(dpc_configure_target tgt)
#  set_target_properties(${tgt} PROPERTIES
#    LANGUAGE CUDA
#  )
#  target_link_options(${tgt} PUBLIC "-cuda;-static-nvidia")
endmacro()

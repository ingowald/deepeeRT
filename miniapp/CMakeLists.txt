# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(OpenMP REQUIRED)


if (OFF)
add_executable(dpMiniApp
  Mesh.h
  Mesh.cpp
  import_OBJ.cpp
  Camera.h
  Camera.cu
  miniapp.cu
  ompShade.cpp
)
target_link_libraries(dpMiniApp PUBLIC deepeeRT_cuda
#  OpenMP::OpenMP_CXX
)
endif()


find_package(VTK)
if (VTK_FOUND)
  add_executable(vtkToDGEF
    vtkToDGEF.cpp
  )
  target_link_libraries(vtkToDGEF PUBLIC
#    deepeeRT
    cuBQL
    VTK::CommonCore
    VTK::IOLegacy
  )
endif()

#    OpenMP::OpenMP_CXX
#    OpenMP::OpenMP_C

#add_executable(bench_cpu_tbb bench.cpp)
#target_link_libraries(bench_cpu_tbb deepee_cpu_tbb)

foreach(config IN ITEMS
    dp_cubql_all_cuda
    dp_cubql_build_cuda_trace_offload
  )
  if (TARGET deepeeRT_${config})
    add_executable(bench_${config} bench.cpp)
    target_link_libraries(bench_${config} deepeeRT_${config})
  endif()
#  add_executable(bench_${config} bench.cpp)
#  target_link_libraries(bench_${config} dp_${config})
endforeach()


#add_executable(bench_cpu_omp bench.cpp)


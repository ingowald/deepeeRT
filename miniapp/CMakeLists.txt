# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(OpenMP REQUIRED)

add_executable(dpMiniApp
  Mesh.h
  Mesh.cpp
  import_OBJ.cpp
  Camera.h
  Camera.cu
  miniapp.cu
  ompShade.cpp
)
target_link_libraries(dpMiniApp PUBLIC deepeeRT
  OpenMP::OpenMP_CXX
)



find_package(VTK)
if (VTK_FOUND)
  add_executable(vtkToDGEF
    vtkToDGEF.cpp
  )
  target_link_libraries(vtkToDGEF PUBLIC
    deepeeRT
    VTK::CommonCore
    VTK::IOLegacy
#    OpenMP::OpenMP_CXX
#    OpenMP::OpenMP_C
  )
endif()



#add_executable(bench_cpu_tbb bench.cpp)
#add_executable(bench_cpu_omp bench.cpp)


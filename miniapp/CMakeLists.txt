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




add_executable(bla
  bla.cpp
)
target_link_libraries(bla PUBLIC 
  OpenMP::OpenMP_CXX
  OpenMP::OpenMP_C
)

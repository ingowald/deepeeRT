# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(CUDAToolkit REQUIRED)

add_library(deepeeRT_common STATIC
  Backend.h
  Backend.cpp
  common.h
  Triangles.h
  Triangles.cpp
  World.h
  World.cpp
  Group.h
  Group.cpp
  Context.h
  Context.cpp
)
target_include_directories(deepeeRT_common PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>/include
#  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_link_libraries(deepeeRT_common PUBLIC
  cuBQL
  CUDA::cuda_driver
  CUDA::cudart_static
)
target_compile_definitions(deepeeRT_common PUBLIC -DdeepeeRT_STATIC=1)

add_library(deepeeRT STATIC
  ../include/deepeeRT/deepeeRT.h
  deepeeRT.cpp
)

add_subdirectory(cuBQL)
target_link_libraries(deepeeRT PUBLIC
  $<BUILD_INTERFACE:deepeeRT_cuBQL>
  $<BUILD_INTERFACE:deepeeRT_common>
)

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(DPC_SOURCE_FILES
)

dpc_configure_source(Camera.cu)
dpc_configure_source(miniapp.cpp)

add_executable(dpMiniApp
  Mesh.h
  Mesh.cpp
  import_OBJ.cpp
  Camera.h
  Camera.cu
  miniapp.cpp
)
dpc_configure_target(dpMiniApp)

target_link_libraries(dpMiniApp PUBLIC deepeeRT deepeeRT_dpc)


// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dp/Triangles.h"
#include "dp/Context.h"

namespace dp {
  TrianglesDP::TrianglesDP(Context         *context,
                           uint64_t         userData,
                           const vec3d     *_vertexArray,
                           int              vertexCount,
                           const vec3i     *_indexArray,
                           int              indexCount)
    : context(context),
      userData(userData),
      vertices(context->device,_vertexArray,vertexCount),
      indices(context->device,_indexArray,indexCount)
  {}
  
  TrianglesDPGroup::TrianglesDPGroup(Context *context,
                                     const std::vector<TrianglesDP *> &meshes)
    : context(context),
      meshes(meshes)
  {}

} // ::dp


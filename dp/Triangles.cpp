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
    : userData(userData),
      vertexArray(_vertexArray),
      indexArray(_indexArray),
      vertexCount(vertexCount),
      indexCount(indexCount),
      context(context)
  {
    assert(vertexArray);
    assert(indexArray);
  }
  
} // ::dp


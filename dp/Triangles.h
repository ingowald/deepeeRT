// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dp/common.h"
#include "dp/DeviceArray.h"
#include "dp/Group.h"

namespace dp {

  struct Context;

  /*! a mesh of triangles, for a dp context, with vertices in doubles */
  struct TrianglesDP {
    TrianglesDP(Context         *context,
                uint64_t         userData,
                const vec3d     *verticArray,
                int              vertexCount,
                const vec3i     *indexArray,
                int              indexCount);
    DeviceArray<vec3d> vertices;
    DeviceArray<vec3i> indices;
    uint64_t     const userData      = 0;
    Context     *const context;
  };

  struct TrianglesDPGroup : public Group {
    TrianglesDPGroup(Context *context,
                     const std::vector<TrianglesDP *> &meshes);
    virtual ~TrianglesDPGroup() = default;

    std::vector<TrianglesDP *> meshes;
    Context *const context;
  };

  
} // ::dp



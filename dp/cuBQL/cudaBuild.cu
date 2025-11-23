// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dp/cuBQL/CuBQLBackend.h"
#include "dp/Context.h"
#include "dp/Group.h"
#include "dp/World.h"

namespace dp_cubql {
  __global__
  void generateTriangleInputs(int meshID,
                              PrimRef *primRefs,
                              box3d *primBounds,
                              int numTrisThisMesh,
                              DevMesh mesh)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numTrisThisMesh) return;

    vec3i idx = mesh.indices[tid];
    box3d bb;
    bb.extend(mesh.vertices[idx.x]);
    bb.extend(mesh.vertices[idx.y]);
    bb.extend(mesh.vertices[idx.z]);

    primRefs[tid] = { meshID, tid };
    primBounds[tid] = bb;
  }
    
}

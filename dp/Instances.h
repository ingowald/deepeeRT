// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dp/Triangles.h"
#include "dp/Ray.h"

namespace dp {

  struct Context;
  struct Group;

  struct InstancesDPGroup {
    InstancesDPGroup(Context *context,
                     const std::vector<Group *> &groups,
                     const affine3d             *d_transforms);
    
    virtual void traceRays(Ray *d_rays, Hit *d_hits, int numRays) = 0;
    
    std::vector<Group *> const groups;
    DeviceArray<affine3d>      transforms;
    Context             *const context;
  };
    
} // ::dp


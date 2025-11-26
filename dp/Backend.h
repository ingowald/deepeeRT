// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dp/common.h"
#include "dp/Triangles.h"
#include "dp/Instances.h"

namespace dp {
  
  struct Context;
  
  /*! implements an actual backend for a double-precision ray tracing
      context. primarily acts as 'factory' for instance and geometry
      groups that then do the actual work */
  struct Backend {
    Backend(Context *const context);
    virtual ~Backend() = default;
    
    virtual std::shared_ptr<InstancesDPGroupImpl>
    createInstancesDPGroupImpl() = 0;
    
    virtual std::shared_ptr<TrianglesDPGroupImpl>
    createTrianglesDPGroupImpl(dp::TrianglesDPGroup *fe) = 0;
    
    Context *const context;
    int const gpuID;
  };
  
} // ::dp

// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dp/cuBQL/trace.h"
#include "dp/cuBQL/TrianglesDP.h"
#include "dp/cuBQL/InstancesDP.h"

namespace dp_cubql {

  __global__ void g_trace(TrianglesDPGroup::DevGroup group,
                          Ray *rays,
                          Hit *hits,
                          int numRays)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    bool dbg = tid == numRays/2;
    if (tid < numRays)
      trace(tid,group,rays,hits,numRays,dbg);
  }
  
  void InstancesDPGroup::traceRays(Ray *rays,
                                   Hit *hits,
                                   int numRays)
  {
    auto device = context->device;
    assert(device->isDevicePointer(rays));
    assert(device->isDevicePointer(hits));

    // we know right now all groups are triangle groups
    TrianglesDPGroup *gg0
      = (TrianglesDPGroup*)groups[0];
    assert(gg0);
    
    int bs = 128;
    int nb = divRoundUp(numRays,bs);
    auto devGroup = gg0->getDevGroup();
    g_trace<<<nb,bs>>>(devGroup,
                       rays,hits,
                       numRays);
    CUBQL_CUDA_SYNC_CHECK();
  }

}

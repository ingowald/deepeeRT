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
  
  void InstancesDPGroup::trace(Ray *rays,
                               Hit *hits,
                               int numRays)
  {
    PING;
    PRINT(numRays);

    dp::Group *fe_group = this->fe->groups[0];
    assert(fe_group);
    
    // we know right now all groups are triangle groups
    dp::TrianglesDPGroup *fe_trianglesGroup
      = (dp::TrianglesDPGroup*)fe_group;
    
    TrianglesDPGroup *my_trianglesGroup
      = (TrianglesDPGroup *)fe_trianglesGroup->impl.get();
    
    // TrianglesDPGroup *triangles,
    //                            Ray *rays,
    //                            Hit *hits,
    //                            int numRays
    int bs = 128;
    int nb = divRoundUp(numRays,bs);
    g_trace<<<nb,bs>>>(my_trianglesGroup->getDevGroup(),
                       rays,hits,
                       numRays);
    CUBQL_CUDA_SYNC_CHECK();
  }

}

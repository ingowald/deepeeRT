// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dp/cuBQL/trace.h"
#include "dp/cuBQL/TrianglesDP.h"

#if DP_TRACE_CUDA
namespace dp_cubql {

  __global__ void g_trace(TrianglesDP::DevGroup group,
                          Ray *rays,
                          Hit *hits,
                          int numRays)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid < numRays)
      trace(tid,group,rays,hits,numRays);
  }
  
  void CuBQLBackend::trace(TrianglesDP *triangles,
                           Ray *rays,
                           Hit *hits,
                           int numRays)
  {
    int bs = 128;
    int nb = divRoundUp(numRays,bs);
    g_trace<<<nb,bs>>>(triangles->getDevGroup(),
                       rays,hits,
                       numRays);
    CUBQL_CUDA_SYNC_CHECK();
  }

}
#endif

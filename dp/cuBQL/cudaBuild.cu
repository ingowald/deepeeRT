// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dp/cuBQL/CuBQLBackend.h"
#include "dp/Context.h"
#include "dp/Group.h"
#include "dp/World.h"

namespace dp_cubql {
  __global__
  void g_generateTriangleInputs(int meshID,
                                PrimRef *primRefs,
                                box3d *primBounds,
                                int numTrisThisMesh,
                                DevMesh mesh)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numTrisThisMesh) return;

    if (tid == 0)
      printf("mesh %p %p\n",
             mesh.vertices,mesh.indices);
    vec3i idx = mesh.indices[tid];
    box3d bb;
    bb.extend(mesh.vertices[idx.x]);
    bb.extend(mesh.vertices[idx.y]);
    bb.extend(mesh.vertices[idx.z]);

    primRefs[tid] = { meshID, tid };
    primBounds[tid] = bb;
  }

  void CuBQLBackend::generateTriangleInputs(int meshID,
                                            PrimRef *primRefs,
                                            box3d *primBounds,
                                            int numTrisThisMesh,
                                            DevMesh mesh)
  {
    PING;
    CUBQL_CUDA_SYNC_CHECK();
    PRINT(primRefs);
    PRINT(primBounds);
    PRINT((int)isDevicePointer(primRefs));
    PRINT((int)isDevicePointer(primBounds));
    int bs = 128;
    int nb = divRoundUp(numTrisThisMesh,bs);
    CUBQL_CUDA_SYNC_CHECK();
    CUBQL_CUDA_SYNC_CHECK_STREAM(0);
    PRINT(primRefs);
    PRINT(primBounds);
    PRINT(numTrisThisMesh);
    PRINT(bs);
    PRINT(nb);
    PRINT(mesh.vertices);
    PRINT(mesh.indices);
    g_generateTriangleInputs<<<nb,bs>>>(meshID,
                                        primRefs,
                                        primBounds,
                                        numTrisThisMesh,
                                        mesh);
    CUBQL_CUDA_SYNC_CHECK();
    CUBQL_CUDA_SYNC_CHECK_STREAM(0);
  }
  
  void CuBQLBackend::bvh_build(bvh_t &bvh,
                               box3d *primBounds,
                               int    numPrims)
  {
    CUBQL_CUDA_SYNC_CHECK();
    CUBQL_CUDA_SYNC_CHECK_STREAM(0);
    
    DeviceMemoryResource memResource;
    PRINT(numPrims);
#if 0
    cuBQL::cuda::radixBuilder(bvh,primBounds,numPrims,
                              BuildConfig(),
                              0,
                              memResource);
#elif 1
    cuBQL::gpuBuilder(bvh,primBounds,numPrims,
                      BuildConfig(),
                      0,
                      memResource);
#else
    cuBQL::cuda::sahBuilder(bvh,primBounds,numPrims,
                            BuildConfig().enableSAH(),
                            0,
                            memResource);
#endif
  }
  
  void CuBQLBackend::bvh_free(bvh_t &bvh)
  {
    DeviceMemoryResource memResource;
    cuBQL::cuda::free(bvh,0,memResource);
  }

  
}

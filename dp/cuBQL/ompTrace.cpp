// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dp/cuBQL/trace.h"

NOT WORKING

namespace dp_cubql {
  using namespace ::cuBQL;
    
  void CuBQLBackend::trace(TrianglesDP *triangles,
                           Ray *rays,
                           Hit *hits,
                           int numRays)
  {
  // void omp_trace(TrianglesDP::DevGroup group,
  //                Ray *rays,
  //                Hit *hits,
  //                int numRays)
  // {
    // int ompNumGPUs = omp_get_num_devices();
    PRINT(dev_deviceCount());
    PRINT(hits);
    // int tid = threadIdx.x+blockIdx.x*blockDim.x;
    // if (tid >= numRays) return;
#pragma omp target is_device_ptr(hits) is_device_ptr(rays)
#pragma omp teams distribute parallel for
    for (int tid=0;tid<numRays;tid++) {
      dp_cubql::trace(tid,
                      triangles->getDevGroup(),
                      rays,hits,numRays);
      // #ifdef NDEBUG
//       const bool dbg = false;
// #else
//       bool dbg = (tid == 1024*512+512);
// #endif

//       Hit hit = hits[tid];
//       hit.primID = -1;
//       int instID = 0;
//       ::cuBQL::ray3d ray(rays[tid].origin,
//                          rays[tid].direction,
//                          rays[tid].tMin,
//                          rays[tid].tMax);
//       if (dbg) {
//         cuBQL::dout << "dbg ray " << ray << "\n";
//         cuBQL::dout << "bvh.nodes " << (int*)group.bvh.nodes << "\n";
//         cuBQL::dout << "bvh.primIDs " << (int*)group.bvh.primIDs << "\n";
//         cuBQL::dout << "group.meshes " << (int*)group.meshes << "\n";
//         cuBQL::dout << "group.mesh0 " << group.meshes[0].userData << "\n";
//         cuBQL::dout << "group.primRefs " << (int*)group.primRefs << "\n";
//       }
//       auto intersectPrim = [&ray,&hit,group,instID,dbg](uint32_t primID) -> double {
//         if (dbg) printf("prim %i\n",primID);
//         RayTriangleIntersection isec;
//         PrimRef prim = group.primRefs[primID];
//         const TriangleDP tri = group.getTriangle(prim);
//         if (isec.compute(ray,tri)) {
//           hit.primID = prim.primID;
//           hit.instID = instID;
//           hit.geomUserData = group.meshes[prim.geomID].userData;
//           hit.t = isec.t;
//           if (dbg) printf("hit %i %i\n",hit.instID,hit.primID);
//           ray.tMax = isec.t;
//         }
//         return ray.tMax;
//       };
//       ::cuBQL::shrinkingRayQuery::forEachPrim(intersectPrim,group.bvh,ray);
//       hits[tid] = hit;
    }
  }
}


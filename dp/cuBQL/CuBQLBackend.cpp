// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dp/cuBQL/CuBQLBackend.h"
#include "dp/Context.h"
#include "dp/Group.h"
#include "dp/World.h"

#include <cuBQL/bvh.h>
#include <cuBQL/math/Ray.h>
#include <cuBQL/queries/triangleData/Triangle.h>
#include <cuBQL/queries/triangleData/math/rayTriangleIntersections.h>
#include <cuBQL/traversal/rayQueries.h>

#include "dp/cuBQL/dpc_cuda.h"
#include "dp/cuBQL/dpc_omp.h"
#include "dp/cuBQL/AutoUploadArray.h"
#include "dp/cuBQL/TrianglesDP.h"

namespace dp_cubql {
  // using namespace ::dp;
  // using namespace ::cuBQL;
    
  // using bvh3d = bvh_t<double,3>;

  using RayTriangleIntersection = cuBQL::RayTriangleIntersection_t<double>;
    
// #if DP_CUDA
//   __global__ void g_trace(TrianglesDP::DevGroup group,
//                           Ray *rays,
//                           Hit *hits,
//                           int numRays)
//   {
//     int tid = threadIdx.x+blockIdx.x*blockDim.x;
//     if (tid >= numRays) return;

// #ifdef NDEBUG
//     const bool dbg = false;
// #else
//     bool dbg = (tid == -1);
// #endif

      
//     Hit hit = hits[tid];
//     hit.primID = -1;
//     int instID = 0;
//     ::cuBQL::ray3d ray(rays[tid].origin,
//                        rays[tid].direction,
//                        rays[tid].tMin,
//                        rays[tid].tMax);

//     if (dbg) {
//       cuBQL::dout << "dbg ray " << ray << "\n";
//       cuBQL::dout << "bvh.nodes " << (int*)group.bvh.nodes << "\n";
//       cuBQL::dout << "bvh.primIDs " << (int*)group.bvh.primIDs << "\n";
//       cuBQL::dout << "group.meshes " << (int*)group.meshes << "\n";
//       cuBQL::dout << "group.mesh0 " << group.meshes[0].userData << "\n";
//       cuBQL::dout << "group.primRefs " << (int*)group.primRefs << "\n";
//     }
//     auto intersectPrim = [&ray,&hit,group,instID,dbg](uint32_t primID) -> double {
//       RayTriangleIntersection isec;
//       PrimRef prim = group.primRefs[primID];
//       const TriangleDP tri = group.getTriangle(prim);
//       if (isec.compute(ray,tri)) {
//         hit.primID = prim.primID;
//         hit.instID = instID;
//         hit.geomUserData = group.meshes[prim.geomID].userData;
//         hit.t = isec.t;
//         if (dbg) printf("hit %i %i\n",hit.instID,hit.primID);
//         ray.tMax = isec.t;
//       }
//       return ray.tMax;
//     };
//     ::cuBQL::shrinkingRayQuery::forEachPrim(intersectPrim,group.bvh,ray);
//     hits[tid] = hit;;
//   }
// #endif

  struct InstancesDP : public dp::InstancesDPImpl {
    InstancesDP(CuBQLBackend *be,
                dp::InstancesDPGroup *fe)
      : InstancesDPImpl(fe), be(be)
    {}
    void trace(Ray *rays,
               Hit *hits,
               int numRays) override;
    CuBQLBackend *const be;
  };


  //     void ompTrace(Ray *rays,
  //                   Hit *hits,
  //                   int numRays)
  //     {
  //       //#pragma omp parallel
  // #pragma omp target
  // #pragma omp teams distribute parallel for
  //       {
  //       }
  //     }
                            
    
  void omp_trace(TrianglesDP::DevGroup group,
                 Ray *rays,
                 Hit *hits,
                 int numRays);
  void cuda_trace(TrianglesDP::DevGroup group,
                  Ray *rays,
                  Hit *hits,
                  int numRays);
    
  void InstancesDP::trace(Ray *rays,
                          Hit *hits,
                          int numRays) 
  {
    be->dev_sync_check();//CUBQL_CUDA_SYNC_CHECK();
    assert(fe->groups.size() == 1);
    assert(fe->d_transforms == nullptr);
    TrianglesDPGroup *tg = (TrianglesDPGroup *)fe->groups[0];
    assert(tg);
    TrianglesDP *triangles = (TrianglesDP*)tg->impl.get();
    assert(triangles);

    int bs = 128;
    int nb = divRoundUp(numRays,bs);
#if DP_OMP
    omp_trace(triangles->getDevGroup(),
              rays,hits,
              numRays);
#endif
#if DP_CUDA
    cuda_trace(triangles->getDevGroup(),
               rays,hits,
               numRays);
#endif
  }
    
  CuBQLBackend::CuBQLBackend(Context *const context)
    : Backend(context)
  {
    // SetActiveGPU forDuration(context->gpuID);
    // cudaFree(0);
    dev_init(context->gpuID);
  }

  std::shared_ptr<dp::InstancesDPImpl>
  CuBQLBackend::createInstancesDPImpl(dp::InstancesDPGroup *fe)
  { return std::make_shared<dp_cubql::InstancesDP>(this,fe); }
    
  std::shared_ptr<dp::TrianglesDPImpl>
  CuBQLBackend::createTrianglesDPImpl(dp::TrianglesDPGroup *fe) 
  { return std::make_shared<dp_cubql::TrianglesDP>(this,fe); }
  
}


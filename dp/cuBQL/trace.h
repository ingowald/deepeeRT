// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// #include "dp/cuBQL/CuBQLBackend.h"
// #include "dp/Context.h"
// #include "dp/Group.h"
// #include "dp/World.h"
#include "dp/cuBQL/TrianglesDP.h"

#include <cuBQL/bvh.h>
#include <cuBQL/math/Ray.h>
#include <cuBQL/queries/triangleData/Triangle.h>
#include <cuBQL/queries/triangleData/math/rayTriangleIntersections.h>
#include <cuBQL/traversal/rayQueries.h>

namespace dp_cubql {
  // using namespace ::cuBQL;
    
  // using bvh3d = bvh_t<double,3>;

  // using TriangleDP = ::cuBQL::triangle_t<double>;
  using RayTriangleIntersection = cuBQL::RayTriangleIntersection_t<double>;

  inline __cubql_both
  void trace(int tid,
             TrianglesDP::DevGroup group,
             Ray *rays,
             Hit *hits,
             int numRays)
  {
    Hit hit = hits[tid];
    hit.primID = -1;
    int instID = 0;

    bool dbg = false;
    
    ::cuBQL::ray3d ray(rays[tid].origin,
                       rays[tid].direction,
                       rays[tid].tMin,
                       rays[tid].tMax);
    if (dbg) {
      cuBQL::dout << "dbg ray " << ray << "\n";
      cuBQL::dout << "bvh.nodes " << (int*)group.bvh.nodes << "\n";
      cuBQL::dout << "bvh.primIDs " << (int*)group.bvh.primIDs << "\n";
      cuBQL::dout << "group.meshes " << (int*)group.meshes << "\n";
      cuBQL::dout << "group.mesh0 " << group.meshes[0].userData << "\n";
      cuBQL::dout << "group.primRefs " << (int*)group.primRefs << "\n";
    }
    auto intersectPrim = [&ray,&hit,group,instID,dbg](uint32_t primID) -> double {
      if (dbg) printf("prim %i\n",primID);
      RayTriangleIntersection isec;
      PrimRef prim = group.primRefs[primID];
      const TriangleDP tri = group.getTriangle(prim);
      if (isec.compute(ray,tri)) {
        hit.primID = prim.primID;
        hit.instID = instID;
        hit.geomUserData = group.meshes[prim.geomID].userData;
        hit.t = isec.t;
        if (dbg) printf("hit %i %i\n",hit.instID,hit.primID);
        ray.tMax = isec.t;
      }
      return ray.tMax;
    };
    ::cuBQL::shrinkingRayQuery::forEachPrim(intersectPrim,group.bvh,ray);
    hits[tid] = hit;
  }

}

// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dp/cuBQL/CuBQLBackend.h"
#include "dp/Context.h"
#include "dp/Group.h"
#if 0
#include <cuBQL/bvh.h>
#include <cuBQL/math/Ray.h>
#include <cuBQL/queries/triangleData/Triangle.h>
#include <cuBQL/queries/triangleData/math/rayTriangleIntersections.h>
#include <cuBQL/traversal/rayQueries.h>

#include "dp/cuBQL/dpc_cuda.h"
#include "dp/cuBQL/dpc_omp.h"
#include "dp/cuBQL/AutoUploadArray.h"
#endif
#include "dp/cuBQL/TrianglesDP.h"
#include "dp/cuBQL/InstancesDP.h"

namespace dp {
  dp::Context *createContext(int gpuID)
  { return new dp_cubql::CuBQLBackend(gpuID); }
  
}

namespace dp_cubql {
  
  CuBQLBackend::CuBQLBackend(int gpuID)
    : dp::Context(gpuID)
  {}

  dp::TrianglesDP *
  CuBQLBackend::createTrianglesDP(/*! a 64-bit user-provided data that
                                    gets attached to this mesh; this is
                                    what gets reported in
                                    Hit::geomUserData if this mesh
                                    yielded the intersection.  */
                                  uint64_t userData,
                                  /*! device array of vertices */
                                  vec3d   *vertexArray,
                                  size_t   vertexCount,
                                  /*! device array of int3 vertex indices */
                                  vec3i   *indexArray,
                                  size_t   indexCount)
  {
    return new TrianglesDP(this,
                           userData,
                           vertexArray,vertexCount,
                           indexArray,indexCount);
  }
    
  
  dp::InstancesDPGroup *
  CuBQLBackend::createInstancesDPGroup(const std::vector<Group *> &instanceGroups,
                                       const affine3d *instanceTransforms)
  {
    PING;
    return new InstancesDPGroup(this,
                                instanceGroups,
                                instanceTransforms);
  }
  
  dp::TrianglesDPGroup *
  CuBQLBackend::createTrianglesDPGroup(const std::vector<dp::TrianglesDP *> &geoms) 
  {
    PING;
    return new TrianglesDPGroup(this,geoms);
  }

}


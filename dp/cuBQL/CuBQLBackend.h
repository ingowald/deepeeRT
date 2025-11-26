// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dp/Context.h"
#include "dp/DeviceAbstraction.h"
#include "dp/Group.h"
#include <cuBQL/bvh.h>

namespace dp_cubql {
  using namespace ::cuBQL;

  using dp::Ray;
  using dp::Hit;
  using dp::Group;
    
  using bvh_t = cuBQL::bvh_t<double,3>;

  struct DevMesh {
    const vec3d   *vertices;
    const vec3i   *indices;
    uint64_t userData;
  };

  // struct TrianglesDPGroup;
  
  struct CuBQLBackend : public dp::Context
  {
    CuBQLBackend(int gpuID);
    virtual ~CuBQLBackend() = default;
    
    dp::TrianglesDP *
    createTrianglesDP(/*! a 64-bit user-provided data that
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
                      size_t   indexCount) override;
    
    dp::TrianglesDPGroup *
    createTrianglesDPGroup(const std::vector<dp::TrianglesDP *> &geoms) override;
    
    dp::InstancesDPGroup *
    createInstancesDPGroup(const std::vector<Group *> &instanceGroups,
                           const affine3d *instanceTransforms) override;
  };

}


  

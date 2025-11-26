// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dp/common.h"
#include "dp/Ray.h"
#include "dp/DeviceAbstraction.h"

namespace dp {

  struct Group;
  struct TrianglesDP;
  struct InstancesDPGroup;
  struct TrianglesDPGroup;
  
  struct Context {
    Context(int gpuID);

    virtual TrianglesDP *
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
                      size_t   indexCount) = 0;
    
    virtual TrianglesDPGroup *
    createTrianglesDPGroup(const std::vector<TrianglesDP *> &geoms) = 0;
    
    virtual InstancesDPGroup *
    createInstancesDPGroup(const std::vector<Group *> &instanceGroups,
                           const affine3d *instanceTransforms) = 0;
    
    /*! the cuda gpu ID that this device is going to run on */
    int const gpuID;
    DeviceAbstraction *device;
  };
  
} // ::dp


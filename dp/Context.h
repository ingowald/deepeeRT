// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dp/common.h"
#include "dp/Backend.h"
#include <memory>

namespace dp {

  struct InstanceGroup;
  struct TriangleMesh;
  struct TrianglesGroup;
  
  struct Context {
    static Context *create(int gpuID);
    
    Context(int gpuID);

    virtual dp::InstanceGroup *
    createInstanceGroup(const std::vector<dp::TrianglesGroup *> &groups,
                        const DPRAffine *transforms) = 0;
    
    virtual dp::TriangleMesh *
    createTriangleMesh(uint64_t         userData,
                       const vec3d     *vertexArray,
                       int              vertexCount,
                       const vec3i     *indexArray,
                       int              indexCount) = 0;
    
    virtual dp::TrianglesGroup *
    createTrianglesGroup(const std::vector<dp::TriangleMesh *> &geoms) = 0;
    
    
    /*! the cuda gpu ID that this device is going to run on */
    int const gpuID;
  };
  
} // ::dp


// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dp/Context.h"
#include <cuBQL/bvh.h>
#include <cuBQL/bvh.h>
#include <cuBQL/math/Ray.h>
#include <cuBQL/queries/triangleData/Triangle.h>
#include <cuBQL/queries/triangleData/math/rayTriangleIntersections.h>
#include <cuBQL/traversal/rayQueries.h>

namespace dp {
  namespace cubql_cuda {
    
    using namespace ::cuBQL;
    
    using bvh3d = bvh_t<double,3>;
    using TriangleDP = cuBQL::triangle_t<double>;
    using RayTriangleIntersection = cuBQL::RayTriangleIntersection_t<double>;
    
    /*! an array that can upload an array from host to device, and free
      on destruction. If the pointer provided is *already* a device
      pointer this will just use that pointer */
    template<typename T>
    struct AutoUploadArray {
      AutoUploadArray(const T *elements, size_t count);
      ~AutoUploadArray() { if (needsCudaFree) cudaFree((void*)elements); }
    
      const T *elements      = 0;
      size_t   count         = 0;
      bool     needsCudaFree = false;
    };
  
    struct CuBQLCUDABackend : public dp::Context
    {
      CuBQLCUDABackend(int gpuID);
      virtual ~CuBQLCUDABackend() = default;

      dp::InstanceGroup *
      createInstanceGroup(const std::vector<dp::TrianglesGroup *> &groups,
                          const DPRAffine *transforms) override;
    
      dp::TriangleMesh *
      createTriangleMesh(uint64_t         userData,
                         const vec3d     *vertexArray,
                         int              vertexCount,
                         const vec3i     *indexArray,
                         int              indexCount) override;
      
      dp::TrianglesGroup *
      createTrianglesGroup(const std::vector<dp::TriangleMesh *> &geoms) override;
    };


    // ==================================================================
    // INLINE IMPLEMENTATION SECTION
    // ==================================================================
    template<typename T> inline
    AutoUploadArray<T>::AutoUploadArray(const T *elements,
                                        size_t count)
    {
      this->count = count;
      if (isDevicePointer(elements)) {
        this->elements = elements;
        this->needsCudaFree = false;
      } else {
        cudaMalloc((void **)&this->elements,count*sizeof(T));
        cudaMemcpy((void*)this->elements,elements,count*sizeof(T),
                   cudaMemcpyDefault);
        this->needsCudaFree = true;
      }
    }
    
  }
}


  

// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dp/Backend.h"
#include "dp/Group.h"
#include <cuBQL/bvh.h>

namespace dp_cubql {
  using namespace ::dp;
  using namespace ::cuBQL;

  using dp::Ray;
    
  using bvh_t = cuBQL::bvh_t<double,3>;
  // using bvh3d = bvh_t<double,3>;

  struct DevMesh {
    const vec3d   *vertices;
    const vec3i   *indices;
    uint64_t userData;
  };

  struct TrianglesDP;
  
  struct CuBQLBackend : public dp::Backend
  {
    CuBQLBackend(Context *const context);
    virtual ~CuBQLBackend() = default;
    
    virtual std::shared_ptr<InstancesDPImpl>
    createInstancesDPImpl(dp::InstancesDPGroup *fe) override;
    
    virtual std::shared_ptr<TrianglesDPImpl>
    createTrianglesDPImpl(dp::TrianglesDPGroup *fe) override;

    // ---------------------- build interface ----------------------
    /*! generate boxes and primrefs for one mesh within a group; all
        data is already allocated */
    void generateTriangleInputs(int meshID,
                                PrimRef *primRefs,
                                box3d *primBounds,
                                int numTrisThisMesh,
                                DevMesh mesh);
    void bvh_build(bvh_t &bvh,
                   box3d *primBounds,
                   int    numPrims);
    void bvh_free(bvh_t &bvh);

    // ---------------------- trace abstraction ----------------------
    void trace(TrianglesDP *dp,
               Ray *rays,
               Hit *hits,
               int numRays);
    
    // ---------------------- device abstraction ----------------------
    /*! device abstraction ... */
    void  dev_init(int devID);
    bool  isDevicePointer(const void *ptr);
    void *dev_malloc(size_t numBytes);
    void  dev_free(void *ptr);
    void  dev_sync_check();
    void  upload(void *devAddr, const void *hostAddr, size_t numBytes);
    void  download(void *hostAddr, const void *devAddr, size_t numBytes);
    int   dev_deviceCount();
  };

}


  

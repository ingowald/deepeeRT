#pragma once

#include "dp/Triangles.h"
#include "dp/cuBQL/CuBQLBackend.h"
#include "dp/cuBQL/AutoUploadArray.h"
#include <cuBQL/bvh.h>
#include <cuBQL/queries/triangleData/Triangle.h>

namespace dp_cubql {
  using dp::Context;
  struct CuBQLBackend;

  using TriangleDP = cuBQL::triangle_t<double>;
  using dp::TrianglesDP;

  /*! allows for referencing a specific primitive within a specific
      geometry within multiple geometries that a group may be built
      over */
  struct PrimRef {
    int geomID;
    int primID;
  };
  
  
  struct TrianglesDPGroup : public dp::TrianglesDPGroup {
    TrianglesDPGroup(Context *context,
                     const std::vector<dp::TrianglesDP *> &geoms);
    virtual ~TrianglesDPGroup();
    
    struct DevGroup {
      inline __cubql_both TriangleDP getTriangle(PrimRef prim) const;
    
      bvh_t     bvh;
      DevMesh  *meshes;
      PrimRef  *primRefs;
    };

    DevGroup getDevGroup() const
    { return { bvh,d_devMeshes,d_primRefs }; }

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

    
    // ---------------------- internal data ----------------------
    bvh_t     bvh = { 0,0,0,0 };
    // DevGroup *group  = nullptr;
    DevMesh  *d_devMeshes = nullptr;
    PrimRef  *d_primRefs  = nullptr;
  };

  inline __cubql_both
  TriangleDP TrianglesDPGroup::DevGroup::getTriangle(PrimRef prim) const
  {
    DevMesh mesh = meshes[prim.geomID];
    vec3i idx = mesh.indices[prim.primID];
    return { mesh.vertices[idx.x],mesh.vertices[idx.y],mesh.vertices[idx.z] };
  }
  
}

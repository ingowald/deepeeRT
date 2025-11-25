#pragma once

#include "dp/cuBQL/CuBQLBackend.h"
#include "dp/cuBQL/AutoUploadArray.h"
#include <cuBQL/bvh.h>
#include <cuBQL/queries/triangleData/Triangle.h>

namespace dp_cubql {
  struct CuBQLBackend;

  using TriangleDP = cuBQL::triangle_t<double>;
  
  /*! triangle mesh representation on the device */
  struct HostMesh {
    HostMesh(CuBQLBackend *be,
             uint64_t userData,
             const vec3d *verticesArray, int verticesCount,
             const vec3i *indicesArray, int indicesCount);
    //   : userData(userData),
    //     vertices(be,verticesArray,verticesCount),
    //     indices(be,indicesArray,indicesCount)
    // {}
    
    DevMesh getDD() const;
    // { return { vertices.elements, indices.elements, userData }; }
    
    AutoUploadArray<vec3d> vertices;
    AutoUploadArray<vec3i> indices;
    uint64_t         const userData;
  };
  
  struct TrianglesDPGroup : public dp::TrianglesDPGroupImpl {
    TrianglesDPGroup(CuBQLBackend *be,
                     dp::TrianglesDPGroup *fe);
    virtual ~TrianglesDPGroup();
    
    struct DevGroup {
      inline __cubql_both TriangleDP getTriangle(PrimRef prim) const;
    
      bvh_t     bvh;
      DevMesh  *meshes;
      PrimRef  *primRefs;
    };

    DevGroup getDevGroup() const
    { return { bvh,meshes,primRefs }; }
      
    bvh_t     bvh;
    // DevGroup *group  = nullptr;
    DevMesh  *meshes = nullptr;
    PrimRef  *primRefs = nullptr;
      
    /*! these are stored and owned on the host, and also manage their
      vertex arrays' ownerhip; but vertex arrays themselves will be
      device accessible */
    std::vector<std::shared_ptr<HostMesh>> hostMeshes;
    
    CuBQLBackend     *const be;
  };

  inline __cubql_both
  TriangleDP TrianglesDPGroup::DevGroup::getTriangle(PrimRef prim) const
  {
    DevMesh mesh = meshes[prim.geomID];
    vec3i idx = mesh.indices[prim.primID];
    return { mesh.vertices[idx.x],mesh.vertices[idx.y],mesh.vertices[idx.z] };
  }
  
}

#include "dp/cuBQL/TrianglesDP.h"
#include "dp/cuBQL/CuBQLBackend.h"

namespace dp_cubql {

  DevMesh getDD(dp::TrianglesDP *mesh) 
  { return { mesh->vertices.data(), mesh->indices.data(), mesh->userData }; }
  
  TrianglesDPGroup::TrianglesDPGroup(Context *context,
                                     const std::vector<dp::TrianglesDP *> &meshes)
    : dp::TrianglesDPGroup(context,meshes)
  {
    auto device = context->device;
    
    int numTrisTotal = 0;
    std::vector<DevMesh> devMeshes;
    for (auto mesh : meshes) {
      numTrisTotal += mesh->indices.size();
      devMeshes.push_back(getDD(mesh));
    }

    d_devMeshes = (DevMesh *)device->malloc(devMeshes.size()*sizeof(DevMesh));
    device->upload(d_devMeshes,devMeshes.data(),devMeshes.size()*sizeof(DevMesh));
      
    d_primRefs = (PrimRef *)device->malloc(numTrisTotal*sizeof(*d_primRefs));
    
    box3d   *primBounds = nullptr;
    primBounds = (box3d *)device->malloc(numTrisTotal*sizeof(*primBounds));

    int offset = 0;
    for (int meshID=0;meshID<(int)meshes.size();meshID++) {
      auto mesh = meshes[meshID];
      int count = mesh->indices.count;
      int bs = 128;
      int nb = divRoundUp(count,bs);
      generateTriangleInputs(meshID,
                             d_primRefs+offset,
                             primBounds+offset,
                             count,
                             getDD(mesh));
      
      offset += count;
    }
    
    std::cout << "#dpr: building BVH over " << prettyNumber(numTrisTotal)
              << " triangles" << std::endl;
    bvh_build(bvh,
              primBounds,
              numTrisTotal);
    std::cout << "#dpr: ... bvh built." << std::endl;
      
    device->free(primBounds);
  }
  
  TrianglesDPGroup::~TrianglesDPGroup()
  {
    auto device = context->device;
    
    device->free(d_devMeshes); 
    device->free(d_primRefs); 
    // ::cuBQL::cuda::free(bvh);
    bvh_free(bvh);
  }

}

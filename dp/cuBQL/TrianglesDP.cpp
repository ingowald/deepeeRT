#include "dp/cuBQL/TrianglesDP.h"
#include "dp/cuBQL/CuBQLBackend.h"

namespace dp_cubql {
  HostMesh::HostMesh(CuBQLBackend *be,
                     uint64_t userData,
                     const vec3d *verticesArray, int verticesCount,
                     const vec3i *indicesArray, int indicesCount)
    : userData(userData),
      vertices(be,verticesArray,verticesCount),
      indices(be,indicesArray,indicesCount)
  {}
    
  DevMesh HostMesh::getDD() const
  { return { vertices.elements, indices.elements, userData }; }
  
  TrianglesDP::TrianglesDP(CuBQLBackend *be,
                           dp::TrianglesDPGroup *fe)
    : TrianglesDPImpl(fe), be(be)
  {
    // SetActiveGPU forDuration(be->gpuID);
      
    int numTrisTotal = 0;
    std::vector<DevMesh> devMeshes;
    for (auto geom : fe->geoms) {
      numTrisTotal += geom->indexCount;
      // this will automatically upload the vertex arrays if so required:
      auto hm = std::make_shared<HostMesh>
        (be,
         geom->userData,
         geom->vertexArray,geom->vertexCount,
         geom->indexArray,geom->indexCount);
      hostMeshes.push_back(hm);
      devMeshes.push_back(hm->getDD());
    }
    // cudaMalloc((void **)&meshes,devMeshes.size()*sizeof(DevMesh));
    meshes = (DevMesh *)be->dev_malloc(devMeshes.size()*sizeof(DevMesh));
    // cudaMemcpy((void*)meshes,devMeshes.data(),
    //            devMeshes.size()*sizeof(DevMesh),cudaMemcpyDefault);
    be->upload(meshes,devMeshes.data(),devMeshes.size()*sizeof(DevMesh));
      
    // cudaMalloc((void **)&primRefs,numTrisTotal*sizeof(*primRefs));
    primRefs = (PrimRef *)be->dev_malloc(numTrisTotal*sizeof(*primRefs));

    box3d   *primBounds = nullptr;
    // cudaMalloc((void **)&primBounds,numTrisTotal*sizeof(*primBounds));
    primBounds = (box3d *)be->dev_malloc(numTrisTotal*sizeof(*primBounds));

    int offset = 0;
    for (int meshID=0;meshID<(int)hostMeshes.size();meshID++) {
      auto &hm = hostMeshes[meshID];
      int count = hm->indices.count;
      int bs = 128;
      int nb = divRoundUp(count,bs);
      // generateTriangleInputs<<<nb,bs>>>(meshID,
      //                                   primRefs+offset,
      //                                   primBounds+offset,
      //                                   count,
      //                                   hm->getDD());
      be->generateTriangleInputs(meshID,
                                 primRefs+offset,
                                 primBounds+offset,
                                 count,
                                 hm->getDD());
      
      offset += count;
    }
    // cudaStreamSynchronize(0);

    std::cout << "#dpr: building BVH over " << prettyNumber(numTrisTotal)
              << " triangles" << std::endl;
    // DeviceMemoryResource memResource;
    // ::cuBQL::cuda::sahBuilder(bvh,
    //                           primBounds,
    //                           numTrisTotal,
    //                           ::cuBQL::BuildConfig(),
    //                           0,
    //                           memResource);
    be->bvh_build(bvh,
                  primBounds,
                  numTrisTotal
                  // ,
                  // ::cuBQL::BuildConfig(),
                  // 0// ,
                  // memResource
                  );
    std::cout << "#dpr: ... bvh built." << std::endl;
      
    // cudaFree(primBounds);
    be->dev_free(primBounds);
  }
  
  TrianglesDP::~TrianglesDP()
  {
    // cudaFree(meshes);
    // cudaFree(primRefs); 
    be->dev_free(meshes); 
    be->dev_free(primRefs); 
    // ::cuBQL::cuda::free(bvh);
    be->bvh_free(bvh);
  }

}

// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "cuBQL/math/box.h"
#include "cuBQL/math/linear.h"
#include "cuBQL/math/affine.h"
#include <cuda_runtime.h>

#include <vector>
#include <map>
#include <memory>
#include <string>
#include <iostream>

namespace dgef {
  using namespace cuBQL;
  
  struct Mesh {
    typedef std::shared_ptr<Mesh> SP;

    void write(std::ostream &out);
    
    std::vector<vec3d>  vertices;
    std::vector<vec3ul> indices;
  };

  struct Instance {
    affine3d transform;
    int      meshID;
  };
  
  struct Model {
    typedef std::shared_ptr<Model> SP;

    static Model::SP load(const std::string &fileName);
    void write(const std::string &fileName);
    
    std::vector<Mesh::SP> meshes;
    std::vector<Instance> instances;
  };

  inline void Mesh::write(std::ostream &out)
  {
    size_t numVertices = vertices.size();
    out.write((char *)&numVertices,sizeof(numVertices));
    out.write((char *)vertices.data(),numVertices*sizeof(numVertices));

    size_t numIndices = indices.size();
    out.write((char *)&numIndices,sizeof(numIndices));
    out.write((char *)indices.data(),numIndices*sizeof(numIndices));
  }
  
  inline void Model::write(const std::string &fileName)
  {
    std::ofstream out(fileName.c_str(),std::ios::binary);
    
    size_t magic = 0x33234567755ull;
    out.write((char *)&magic,sizeof(magic));
    size_t numMeshes = meshes.size();
    out.write((char *)&numMeshes,sizeof(numMeshes));
    for (auto mesh : meshes)
      mesh->write(out);
    size_t numInstances = instances.size();
    out.write((char *)&numInstances,sizeof(numInstances));
    out.write((char *)instances.data(),numInstances*sizeof(instances[0]));
  }
  
}


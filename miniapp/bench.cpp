#include "deepee/deepee.h"
#include "DGEF.h"
#include "dp/DeviceAbstraction.h"
#include <fstream>

void usage(const std::string &err="")
{
  std::cout
    << "usage: ./bench"
    " -r <rays>"
    " -m <model>"
    << std::endl;
  if (!err.empty()) 
    throw std::runtime_error(err);
  exit(0);
}

template<typename T>
std::vector<T> loadVectorOf(std::ifstream &in, int part=0, int numParts=1)
{
  if (!in.good()) throw std::runtime_error("invalid input stream");
  in.seekg(0,std::ios::end);
  size_t size = in.tellg();
  in.seekg(0,std::ios::beg);
  size_t count = size/sizeof(T);
  std::vector<T> vec;
  size_t begin = part * count / numParts;
  size_t end = (part+1) * count / numParts;
  T t;
  for (size_t i=0;i<begin;i++)
    in.read((char *)&t,sizeof(t));
  for (size_t i=begin;i<end;i++) {
    in.read((char *)&t,sizeof(t));
    vec.push_back(t);
  }
  for (size_t i=end;i<count;i++)
    in.read((char *)&t,sizeof(t));
  return vec;
}

template<typename T>
std::vector<T> loadVectorOf(const std::string &fn, int part=0, int numParts=1)
{
  std::ifstream in(fn,std::ios::binary);
  return loadVectorOf<T>(in,part,numParts);
}

int main(int ac, char **av)
{
  std::string raysFileName;
  std::string modelFileName;
  int gpuID = 0;
  for (int i=1;i<ac;i++) {
    std::string arg = av[i];
    if (arg == "-r")
      raysFileName = av[++i];
    else if (arg == "-m")
      modelFileName = av[++i];
    else {
      usage("unknown argument '"+arg+"'");
    }
  }

  if (modelFileName.empty()) usage("no model file specified");
  if (raysFileName.empty()) usage("no ray file specified");

  dgef::Model::SP model = dgef::Model::read(modelFileName);
  if (model->meshes.size() != 1) throw std::runtime_error("unsupported config...");
  if (model->instances.size() != 1) throw std::runtime_error("unsupported config...");

  dp::DeviceAbstraction *device = new dp::DeviceAbstraction(gpuID);
  DPRContext dpc = dprContextCreate(DPR_CONTEXT_GPU,gpuID);
  std::vector<DPRTriangles> geoms;
  for (auto mesh : model->meshes) {
    std::vector<cuBQL::vec3i> int_indices;
    for (auto idx : mesh->indices)
      int_indices.push_back({(int)idx.x,(int)idx.y,(int)idx.z});
    geoms.push_back(dprCreateTrianglesDP(dpc,
                                         geoms.size(),
                                         (DPRvec3*)mesh->vertices.data(),
                                         mesh->vertices.size(),
                                         (DPRint3*)int_indices.data(),
                                         mesh->indices.size()));
  }
  DPRGroup group
    = dprCreateTrianglesGroup(dpc,geoms.data(),geoms.size());
  DPRWorld world
    = dprCreateWorldDP(dpc,&group,nullptr,1);

  std::vector<DPRRay> h_rays = loadVectorOf<DPRRay>(raysFileName);
  int numRays = h_rays.size();

  DPRRay *d_rays = (DPRRay*)device->malloc(numRays*sizeof(DPRRay));
  device->upload(d_rays,h_rays.data(),numRays*sizeof(DPRRay));
  DPRHit *d_hits = (DPRHit*)device->malloc(numRays*sizeof(DPRHit));
  device->syncCheck();

  dprTrace(world,d_rays,d_hits,numRays);
  
}

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

    inline double getCurrentTime()
    {
#ifdef _WIN32
      SYSTEMTIME tp; GetSystemTime(&tp);
      /*
         Please note: we are not handling the "leap year" issue.
     */
      size_t numSecsSince2020
          = tp.wSecond
          + (60ull) * tp.wMinute
          + (60ull * 60ull) * tp.wHour
          + (60ull * 60ul * 24ull) * tp.wDay
          + (60ull * 60ul * 24ull * 365ull) * (tp.wYear - 2020);
      return double(numSecsSince2020 + tp.wMilliseconds * 1e-3);
#else
      struct timeval tp; gettimeofday(&tp,nullptr);
      return double(tp.tv_sec) + double(tp.tv_usec)/1E6;
#endif
    }

#ifdef __WIN32__
#  define osp_snprintf sprintf_s
#else
#  define osp_snprintf snprintf
#endif
  

    /*! added pretty-print function for large numbers, printing 10000000 as "10M" instead */
    inline std::string prettyDouble(const double val) {
      const double absVal = abs(val);
      char result[1000];

      if      (absVal >= 1e+18f) osp_snprintf(result,1000,"%.1f%c",float(val/1e18f),'E');
      else if (absVal >= 1e+15f) osp_snprintf(result,1000,"%.1f%c",float(val/1e15f),'P');
      else if (absVal >= 1e+12f) osp_snprintf(result,1000,"%.1f%c",float(val/1e12f),'T');
      else if (absVal >= 1e+09f) osp_snprintf(result,1000,"%.1f%c",float(val/1e09f),'G');
      else if (absVal >= 1e+06f) osp_snprintf(result,1000,"%.1f%c",float(val/1e06f),'M');
      else if (absVal >= 1e+03f) osp_snprintf(result,1000,"%.1f%c",float(val/1e03f),'k');
      else if (absVal <= 1e-12f) osp_snprintf(result,1000,"%.1f%c",float(val*1e15f),'f');
      else if (absVal <= 1e-09f) osp_snprintf(result,1000,"%.1f%c",float(val*1e12f),'p');
      else if (absVal <= 1e-06f) osp_snprintf(result,1000,"%.1f%c",float(val*1e09f),'n');
      else if (absVal <= 1e-03f) osp_snprintf(result,1000,"%.1f%c",float(val*1e06f),'u');
      else if (absVal <= 1e-00f) osp_snprintf(result,1000,"%.1f%c",float(val*1e03f),'m');
      else osp_snprintf(result,1000,"%f",(float)val);

      return result;
    }
  

    /*! return a nicely formatted number as in "3.4M" instead of
        "3400000", etc, using mulitples of thousands (K), millions
        (M), etc. Ie, the value 64000 would be returned as 64K, and
        65536 would be 65.5K */
    inline std::string prettyNumber(const size_t s)
    {
      char buf[1000];
      if (s >= (1000LL*1000LL*1000LL*1000LL)) {
        osp_snprintf(buf, 1000,"%.2fT",s/(1000.f*1000.f*1000.f*1000.f));
      } else if (s >= (1000LL*1000LL*1000LL)) {
        osp_snprintf(buf, 1000, "%.2fG",s/(1000.f*1000.f*1000.f));
      } else if (s >= (1000LL*1000LL)) {
        osp_snprintf(buf, 1000, "%.2fM",s/(1000.f*1000.f));
      } else if (s >= (1000LL)) {
        osp_snprintf(buf, 1000, "%.2fK",s/(1000.f));
      } else {
        osp_snprintf(buf,1000,"%zi",s);
      }
      return buf;
    }

    /*! return a nicely formatted number as in "3.4M" instead of
        "3400000", etc, using mulitples of 1024 as in kilobytes,
        etc. Ie, the value 65534 would be 64K, 64000 would be 63.8K */
    inline std::string prettyBytes(const size_t s)
    {
      char buf[1000];
      if (s >= (1024LL*1024LL*1024LL*1024LL)) {
        osp_snprintf(buf, 1000,"%.2fT",s/(1024.f*1024.f*1024.f*1024.f));
      } else if (s >= (1024LL*1024LL*1024LL)) {
        osp_snprintf(buf, 1000, "%.2fG",s/(1024.f*1024.f*1024.f));
      } else if (s >= (1024LL*1024LL)) {
        osp_snprintf(buf, 1000, "%.2fM",s/(1024.f*1024.f));
      } else if (s >= (1024LL)) {
        osp_snprintf(buf, 1000, "%.2fK",s/(1024.f));
      } else {
        osp_snprintf(buf,1000,"%zi",s);
      }
      return buf;
    }
  

int main(int ac, char **av)
{
  using cuBQL::vec3d;
  using cuBQL::vec3i;
  
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
    PING;

    int numVertices = mesh->vertices.size();
    int numIndices = mesh->indices.size();
    vec3d *d_vertices = (vec3d *)device->malloc(numVertices*sizeof(vec3d));
    vec3i *d_indices = (vec3i *)device->malloc(numIndices*sizeof(vec3i));
    device->upload(d_vertices,mesh->vertices.data(),numVertices*sizeof(vec3d));
    device->upload(d_indices,int_indices.data(),numIndices*sizeof(vec3i));
    geoms.push_back(dprCreateTrianglesDP(dpc,
                                         geoms.size(),
                                         (DPRvec3*)d_vertices,
                                         mesh->vertices.size(),
                                         (DPRint3*)d_indices,
                                         mesh->indices.size()));
  }
  DPRGroup group
    = dprCreateTrianglesGroup(dpc,geoms.data(),geoms.size());
  DPRWorld world
    = dprCreateWorldDP(dpc,&group,nullptr,1);

  std::vector<DPRRay> h_rays = loadVectorOf<DPRRay>(raysFileName);
  int numRays = h_rays.size();

  PING;
  
  DPRRay *d_rays = (DPRRay*)device->malloc(numRays*sizeof(DPRRay));
  
  device->upload(d_rays,h_rays.data(),numRays*sizeof(DPRRay));
  PING;
  DPRHit *d_hits = (DPRHit*)device->malloc(numRays*sizeof(DPRHit));
  device->syncCheck();
  PING;

  double targetNumSeconds = 1.f;
  double t0 = getCurrentTime();
  double numSecsTaken = 0;
  int numRunsDone = 0;
  int numRunsPerBatch = 1;
  while (true) {
    for (int it=0;it<numRunsPerBatch;it++) {
      dprTrace(world,d_rays,d_hits,numRays);
      numRunsDone++;
    }
    numRunsPerBatch *= 2;
    numSecsTaken = getCurrentTime() - t0;
    if (numSecsTaken >= targetNumSeconds)
      break;
  }
  std::cout << "done tracing.... done " << numRunsDone << " runs in "
            << numSecsTaken << " seconds" << std::endl;
  std::cout << "that's a total of "
            << prettyNumber(numRays*double(numRunsDone)/numSecsTaken)
            << " rays per second" << std::endl;
}

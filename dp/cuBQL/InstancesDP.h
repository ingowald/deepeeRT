#pragma once

#include "dp/Instances.h"

namespace dp_cubql {

  struct CuBQLBackend;
  
  struct InstancesDPGroup : public dp::InstancesDPGroup {
    InstancesDPGroup(Context *context,
                     const std::vector<Group *> &groups,
                     const affine3d             *d_transforms)
      : dp::InstancesDPGroup(context,groups,d_transforms)
    { }
    
    void traceRays(Ray *d_rays, Hit *d_hits, int numRays) override;
  };

}

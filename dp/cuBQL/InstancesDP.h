#pragma once

#include "dp/Backend.h"
#include "dp/World.h"

namespace dp_cubql {

  struct CuBQLBackend;
  
  struct InstancesDPGroup : public dp::InstancesDPGroupImpl {
    InstancesDPGroup(CuBQLBackend *be,
                     dp::InstancesDPGroup *fe)
      : InstancesDPGroupImpl(fe), be(be)
    { assert(fe); assert(be); }
    
    void trace(Ray *rays,
               Hit *hits,
               int numRays) override;
    CuBQLBackend *const be;
  };

}

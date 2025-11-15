// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "Mesh.h"
#include "Camera.h"
#include <fstream>
#include <omp.h>

namespace miniapp {

  /*! helper function that creates a semi-random color from an ID */
  inline __cubql_both vec3f randomColor(int i)
  {
    const uint64_t FNV_offset_basis = 0xcbf29ce484222325ULL;
    const uint64_t FNV_prime = 0x10001a7;
    uint32_t v = (uint32_t)FNV_offset_basis;
    v = FNV_prime * v ^ i;
    v = FNV_prime * v ^ i;
    v = FNV_prime * v ^ i;
    v = FNV_prime * v ^ i;

    int r = v >> 24;
    v = FNV_prime * v ^ i;
    int b = v >> 16;
    v = FNV_prime * v ^ i;
    int g = v >> 8;
    return vec3f((r&255)/255.f,
                 (g&255)/255.f,
                 (b&255)/255.f);
  }

  void ompShade(vec4f *d_pixels,
                DPRRay *d_rays,
                DPRHit *d_hits,
                const vec2i fbSize)
  {
    //#pragma omp parallel
// #pragma omp target
#pragma omp target is_device_ptr(d_pixels) is_device_ptr(d_hits) is_device_ptr(d_rays)
#pragma omp teams distribute parallel for
    for (int tid=0;tid<fbSize.x*fbSize.y;tid++) {
      DPRHit hit = d_hits[tid];
      vec3f color = randomColor(hit.primID + 0x290374*hit.geomUserData);
      vec4f pixel = {color.x,color.y,color.z,1.f};
      d_pixels[tid] = pixel;
    }
  }
                            
    
}

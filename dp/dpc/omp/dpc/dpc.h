// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>
#include <omp.h>

#define DPC_OMP 1
#define DPC_OPENMP 1

namespace dpc {
  struct uint2 { uint32_t x, y; };
  
  struct Device {
    Device(int gpuID) {
      int numDevices = omp_get_num_devices();
      if (numDevices == 0)
        throw std::runtime_error("cannot find openmp offload device(s)");
      PRINT(numDevices);
      int initDevice = omp_get_initial_device();
      PRINT(initDevice);
      // cudaSetDevice(gpuID); cudaFree(0);
    }
    void syncCheck()
    { cudaStreamSynchronize(0); }
    void download(void *h_ptr, const void *d_ptr, size_t sz)
    { cudaMemcpy(h_ptr,d_ptr,sz,cudaMemcpyDefault); }
    void upload(void *d_ptr, const void *h_ptr, size_t sz)
    { cudaMemcpy(d_ptr,h_ptr,sz,cudaMemcpyDefault); }
    void malloc(void **ptr, size_t sz)
    {
      PING;
      cudaMalloc(ptr,sz);
      PRINT(ptr);
      *ptr = omp_target_alloc(sz,0);
      PRINT(ptr);
    }
  };

  struct Kernel2D {
    inline Kernel2D(uint32_t nbx, uint32_t nby, uint32_t bsx, uint32_t bsy,
                    size_t tid);
    inline uint2 workIdx() const;

    uint2 ws, wi;
  };


#pragma omp declare target
  inline Kernel2D::Kernel2D(uint32_t nbx, uint32_t nby,
                            uint32_t bsx, uint32_t bsy,
                            size_t tid)
    {
      ws.x = nbx * bsx;
      ws.y = nby * bsy;
      wi.y = uint32_t(tid / (size_t)ws.x);
      wi.x = tid - wi.y*ws.x;
    };
#pragma omp end declare target

#pragma omp declare target
  inline uint2 Kernel2D::workIdx() const
  { return wi; }
#pragma omp end declare target

}



#define __dpc_device /* inline */

#define DPC_KERNEL(kernelName) inline void kernelName

#define DPC_KERNEL2D_CALL(dpcDevice,kernelName,_nb,_bs,...)    \
  {                                                                     \
    size_t numTotal = (_nb.x*(size_t)_bs.x*_nb.y*(size_t)_nb.y);        \
    _Pragma("omp target device(0)") {                                   \
      for (size_t i=0;i<numTotal;i++)                                   \
        kernelName(dpc::Kernel2D(_bs.x,_bs.y,_bs.x,_bs.y,i),__VA_ARGS__); \
    }                                                                   \
  }


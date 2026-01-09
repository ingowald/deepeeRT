// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>

#define DPC_CUDA 1

#define DPC_KERNEL(kernelName) __global__ void kernelName

#define DPC_KERNEL_CALL(dpcDevice,kernelName,nb,bs,...)    \
  {                                                             \
    kernelName<<<(uint32_t)nb,(uint32_t)bs>>>(__VA_ARGS__);               \
  }

#define DPC_KERNEL2D_CALL(dpcDevice,kernelName,_nb,_bs,...)     \
  {                                                             \
    kernelName<<<nb,bs>>>(dpc::Kernel2D{},__VA_ARGS__);         \
  }

#define __dpc_device __device__

namespace dpc {
  struct uint2 { uint32_t x, y; };
  
#ifdef __CUDACC__
  struct Kernel2D {
    inline __device__
    Kernel2D()
    {};

    inline __device__
    uint2 workIdx() const
    { return
        {
          threadIdx.x+blockIdx.x*blockDim.x,
          threadIdx.y+blockIdx.y*blockDim.y
        };
    }
    
  };
#endif
  struct Device {
    Device(int gpuID) { cudaSetDevice(gpuID); cudaFree(0); }
    void syncCheck()
    { cudaStreamSynchronize(0); }
    void download(void *h_ptr, const void *d_ptr, size_t sz)
    { cudaMemcpy(h_ptr,d_ptr,sz,cudaMemcpyDefault); }
    void upload(void *d_ptr, const void *h_ptr, size_t sz)
    { cudaMemcpy(d_ptr,h_ptr,sz,cudaMemcpyDefault); }
    void malloc(void **ptr, size_t sz)
    { cudaMalloc(ptr,sz); }
  };
  
};

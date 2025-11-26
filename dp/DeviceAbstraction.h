#pragma once

#include "dp/common.h"

namespace dp {
  
  struct DeviceAbstraction {
    DeviceAbstraction(int gpuID);
    virtual ~DeviceAbstraction();

    static bool  isDevicePointer(const void *ptr);
    void *malloc(size_t numBytes);
    void  free(void *ptr);
    void  syncCheck();
    void  upload(void *devAddr, const void *hostAddr, size_t numBytes);
    void  download(void *hostAddr, const void *devAddr, size_t numBytes);
    static int getDeviceCount();

    int const gpuID;
  };
  
}

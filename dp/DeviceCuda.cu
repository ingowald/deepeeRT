#include "dp/DeviceAbstraction.h"
#include "cuBQL/bvh.h"

namespace dp {

    /*! helper class that sets the active cuda device to the given gpuID
      for the lifetime of this class, and restores it to whatever it
      was after that variable dies */
  struct SetActiveGPU {
    SetActiveGPU(int gpuID) { cudaGetDevice(&savedActive); cudaSetDevice(gpuID); }
    ~SetActiveGPU() { cudaSetDevice(savedActive); }
    int savedActive = -1;
  };

  DeviceAbstraction::DeviceAbstraction(int gpuID)
    : gpuID(gpuID)
  {
    SetActiveGPU forDuration(gpuID);
    cudaFree(0);
    syncCheck();
  }
  
  DeviceAbstraction::~DeviceAbstraction()
  {
  }
  

  bool  DeviceAbstraction::isDevicePointer(const void *ptr)
  {
    cudaPointerAttributes attributes = {};
    // do NOT check for error: in CUDA<10, passing a host pointer will
    // actually create a cuda error, so let's just call and then
    // ignore/clear the error
    CUBQL_CUDA_CALL(PointerGetAttributes(&attributes,(const void *)ptr));
    // cudaPointerGetAttributes(&attributes,(const void *)ptr);
    // PRINT((int)attributes.type);
    // PRINT((int)cudaMemoryTypeHost);
    cudaGetLastError();

    bool isDevice
      = 
      attributes.type == cudaMemoryTypeManaged
      ||
      attributes.type == cudaMemoryTypeDevice;
    // PRINT((int)isDevice);
    return isDevice;
  }
  
  void *DeviceAbstraction::malloc(size_t numBytes)
  {
    SetActiveGPU forDuration(gpuID);
    void *ptr = 0;
    CUBQL_CUDA_CALL(Malloc(&ptr,numBytes));
    CUBQL_CUDA_SYNC_CHECK();
    PRINT(numBytes);
    PING; PRINT(ptr);
    return ptr;
  }
  
  void  DeviceAbstraction::free(void *ptr)
  {
    SetActiveGPU forDuration(gpuID);
    CUBQL_CUDA_CALL(Free(ptr));
  }
  
  void  DeviceAbstraction::syncCheck()
  {
    SetActiveGPU forDuration(gpuID);
    CUBQL_CUDA_SYNC_CHECK();
  }
  
  void  DeviceAbstraction::upload(void *devAddr, const void *hostAddr, size_t numBytes)
  {
    PING;
    CUBQL_CUDA_SYNC_CHECK();
    PRINT(devAddr);
    PRINT(hostAddr);
    PRINT(numBytes);
    PRINT((int)isDevicePointer(devAddr));
    PRINT((int)isDevicePointer(hostAddr));
    assert(isDevicePointer(devAddr));
    assert(!isDevicePointer(hostAddr));
    SetActiveGPU forDuration(gpuID);
    CUBQL_CUDA_CALL(Memcpy(devAddr,hostAddr,numBytes,cudaMemcpyDefault));
  }
  
  void  DeviceAbstraction::download(void *hostAddr, const void *devAddr, size_t numBytes)
  {
    SetActiveGPU forDuration(gpuID);
    CUBQL_CUDA_CALL(Memcpy(hostAddr,devAddr,numBytes,cudaMemcpyDefault));
  }
  
  int DeviceAbstraction::getDeviceCount()
  {
    int gpuCount = 0;
    CUBQL_CUDA_CALL(GetDeviceCount(&gpuCount));
    CUBQL_CUDA_SYNC_CHECK();
    return gpuCount;
  }
  
}

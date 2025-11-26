#pragma once

#include "dp/DeviceAbstraction.h"

namespace dp {

  template<typename T>
  struct DeviceArray {
    DeviceArray(DeviceAbstraction *device,
                 const T *pData,
                 size_t   count);
    virtual ~DeviceArray();
    
    T *data() { return d_data; }
    size_t size() { return count; }
    
    T     *d_data;
    size_t count;
    DeviceAbstraction *const device;
  };

  template<typename T>
  inline DeviceArray<T>::DeviceArray(DeviceAbstraction *device,
                                     const T *pData,
                                     size_t   _count)
    : device(device),
      count(_count)
  {
    d_data
      = count
      ? (T*)device->malloc(count*sizeof(T))
      : 0;
    device->upload(d_data,pData,count*sizeof(T));
  }

  template<typename T>
  inline DeviceArray<T>::~DeviceArray()
  {
    if (d_data) device->free((void *)d_data);
  }

}

  

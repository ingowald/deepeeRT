// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dp/cuBQL/CuBQLBackend.h"

namespace dp_cubql {
  using namespace ::dp;
  
  /*! an array that can upload an array from host to device, and free
    on destruction. If the pointer provided is *already* a device
    pointer this will just use that pointer */
  template<typename T>
  struct AutoUploadArray {
    AutoUploadArray(CuBQLBackend *be, const T *elements, int count)
      : be(be)
    {
      this->count = count;
      if (be->isDevicePointer(elements)) {
        this->elements = elements;
        this->needsFree = false;
      } else {
        this->elements = (T *)be->dev_malloc(count*sizeof(T));
        be->upload((void *)this->elements,(const void *)elements,count*sizeof(T));
        this->needsFree = true;
      }
    }
      
    ~AutoUploadArray() { if (needsFree) be->dev_free((void*)elements); }
    
    const T *elements  = 0;
    int  count         = 0;
    bool needsFree = false;
    CuBQLBackend *const be;
  };

  
}

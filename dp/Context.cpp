// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dp/Context.h"
#include "dp/cuBQL/CuBQLBackend.h"

namespace dp {
  
  Context::Context(int gpuID)
    : gpuID(gpuID)
  {}

  Context *Context::create(int gpuID)
  {
    return new cubql_cuda::CuBQLCUDABackend(gpuID);
  };
  
} // ::dp


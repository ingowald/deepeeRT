// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "deepee/deepee.h"
#include "cuBQL/math/box.h"
#include "cuBQL/math/linear.h"
#include "cuBQL/math/affine.h"
#if DEEPEE_CUDA
# include <cuda_runtime.h>
#endif

#include <vector>
#include <map>
#include <memory>
#include <string>
#include <iostream>

namespace dp {
  using namespace ::cuBQL;

  using cuBQL::vec3d;
  using cuBQL::vec3i;
  using cuBQL::affine3d;
  
}

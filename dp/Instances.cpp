// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dp/Instances.h"
#include "dp/Context.h"

namespace dp {

  InstancesDPGroup::InstancesDPGroup(Context *context,
                                     const std::vector<Group *> &groups,
                                     const affine3d             *d_transforms)
    : context(context),
      groups(groups),
      transforms(context->device,
                 d_transforms,
                 d_transforms ? groups.size() : 0)
  {}
  
} // ::dp


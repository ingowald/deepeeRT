# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(CUDAToolkit)# REQUIRED)
if (CUDA_FOUND)
  message("found CUDA, enabling cuda builder and traversers")
  set(DP_HAVE_CUDA ON)
else()
  message("CUDA not found - disabling")
endif()

if (DP_HAVE_OFFLOAD)
  find_package(OpenMP REQUIRED)
#  find_package(OpenMP)
#  if (OpenMP_FOUND)
#    message("found OpenMP, enabling cuda builder and traversers")
#    set(DP_HAVE_OFFLOAD ON)
#  else()
#    message("OpenMP not found - disabling")
#  endif()
endif()

add_library(dp_config INTERFACE)
target_include_directories(dp_config INTERFACE
  ${PROJECT_SOURCE_DIR}
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_link_libraries(dp_config INTERFACE
  cuBQL
#  OpenMP::OpenMP_CXX
)

set(DEEPEE_SOURCES
  ../include/deepee/deepee.h
  common.h
  Triangles.h
  Triangles.cpp
  Instances.h
  Instances.cpp
  Context.h
  Context.cpp
  Group.h
  Group.cpp
  deepeeRT.cpp
  
  cuBQL/CuBQLBackend.h
  cuBQL/CuBQLBackend.cpp
  cuBQL/TrianglesDP.h
  cuBQL/TrianglesDP.cpp
)


# ==================================================================
# *DEVICE*: general malloc/free/memcpy/etc device abstraction
# can be consumed by a traverser
# ==================================================================
if (DP_HAVE_CUDA)
  add_library(dp_device_cuda DeviceCuda.cu)
  target_link_libraries(dp_device_cuda PUBLIC
    dp_config
    CUDA::cuda_driver
    CUDA::cudart_static
  )
endif()

if (DP_HAVE_OFFLOAD)
  add_library(dp_device_offload DeviceOffload.cpp)
  target_link_libraries(dp_device_offload PUBLIC
    dp_config
    OpenMP::OpenMP_CXX
  )
endif()

# ==================================================================
# *UPLOAD*: how to get a BVH from whereever it is to a device where it
# can be consumed by a traverser
# ==================================================================
if (DP_HAVE_CUDA)
  set(TGT dp_cubql_upload_cuda)
  add_library(${TGT} cuBQL/upload_via_cuda.cu)
  target_link_libraries(${TGT} PUBLIC
    dp_config
    dp_device_cuda
#    CUDA::cuda_driver
#    CUDA::cudart_static
  )
endif()

if (DP_HAVE_OFFLOAD)
  set(TGT dp_cubql_upload_offload)
  add_library(${TGT} cuBQL/upload_via_offload.cpp)
  target_link_libraries(${TGT} PUBLIC
    dp_device_offload
#    dp_config
#    OpenMP::OpenMP_CXX
  )
endif()

if (ON) # always available
  set(TGT dp_cubql_upload_ignore)
  add_library(${TGT} cuBQL/upload_ignore.cpp)
  target_link_libraries(${TGT} PUBLIC
    dp_config
  )
endif()

# ==================================================================
# *BUILD*: depends on where the result will need to go
# ==================================================================

# ------------------------------------------------------------------
# - use cuBQL cuda/gpu builders
# - consumer of gpu is cuda (leave on gpu, nothing to do)
# ------------------------------------------------------------------
if (DP_HAVE_CUDA)
  # doesn't need a "bulid_*for*" because this'll always be on gpu
  set(TGT dp_cubql_build_cuda)
  add_library(${TGT} cuBQL/cudaBuild.cu)
  #  target_compile_definitions(${TGT} PUBLIC)
  target_link_libraries(${TGT} PUBLIC
    dp_config
    $<BUILD_INTERFACE:cuBQL_cuda_double3_static>
  )
endif()
  
# ------------------------------------------------------------------
# - use cuBQL *HOST* builders
# - consumer of gpu is omp target -> upload through omp
# ------------------------------------------------------------------
if (ON) # always have host builders...
  set(TGT dp_cubql_build_cpu)
  add_library(${TGT} cuBQL/cpuBuild.cpp)
  target_compile_definitions(${TGT} PUBLIC)
  target_link_libraries(${TGT} PUBLIC
    dp_config
  )
endif()

# ==================================================================
# *TRACE* implementation
# ==================================================================
if (DP_HAVE_CUDA)
  set(TGT dp_cubql_trace_cuda)
  add_library(${TGT} STATIC ${DEEPEE_SOURCES}
    cuBQL/cudaTrace.cu
  )
#  target_compile_definitions(${TGT} PUBLIC -DDP_TRACE_CUDA=1)
  target_link_libraries(${TGT} PUBLIC
    dp_config
  )
endif()

if (DP_HAVE_OFFLOAD)
  set(TGT dp_cubql_trace_offload)
  add_library(${TGT} STATIC ${DEEPEE_SOURCES}
    cuBQL/ompTrace.cpp
  )
#  target_compile_definitions(${TGT} PUBLIC -DDP_TRACE_OFFLOAD=1)
  target_link_libraries(${TGT} PUBLIC
    dp_config
  )
endif()


# ==================================================================
# *FINAL* deepeeRT realizations
# ==================================================================

add_library(dp_common STATIC ${DEEPEE_SOURCES})
target_link_libraries(dp_common PUBLIC dp_config)

# - build cuBQL BVH with cuda builders (on device, leave on device)
# - traverse with cuda cuBQL traveral
if (DP_HAVE_CUDA)
  set(TGT dp_cubql_all_cuda)
  message("adding interface ${TGT}")
  add_library(${TGT} INTERFACE)
  target_link_libraries(${TGT} INTERFACE
    dp_common
    dp_device_cuda
    dp_cubql_build_cuda
    dp_cubql_upload_ignore
    dp_cubql_trace_cuda
  )

  if (TARGET ${TGT})
    message("${TGT} exists")
  else()
    message("${TGT} does NOT exist")
  endif()
endif()

# - build cuBQL BVH with *cpu* builders (on host, then upload via cuda)
# - traverse with cuda cuBQL traveral
if (DP_HAVE_CUDA)
  set(TGT dp_cubql_build_cpu_trace_cuda)
  add_library(${TGT} INTERFACE)
  target_link_libraries(${TGT} INTERFACE
    dp_common
    dp_cubql_build_cpu
    dp_cubql_upload_cuda
    dp_cubql_trace_cuda)
endif()

# - build on cpu, then upload to device (via openmp)
# - trace with openmp on device
if (DP_HAVE_OFFLOAD)
  set(TGT dp_cubql_build_cpu_trace_offload)
  add_library(${TGT} INTERFACE)
  target_link_libraries(${TGT} INTERFACE
    dp_common
    dp_cubql_build_cpu
    dp_cubql_upload_offload
    dp_cubql_trace_offload)
endif()

# - build on gpu, with cuda builders
# - trace with openmp on device
if (DP_HAVE_OFFLOAD AND DP_HAVE_CUDA)
  set(TGT dp_cubql_build_cuda_trace_offload)
  add_library(${TGT} INTERFACE)
  target_link_libraries(${TGT} INTERFACE
    dp_common
    dp_cubql_build_cuda
    dp_cubql_upload_ignore
    dp_cubql_trace_offload)
endif()


# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
cmake_policy(SET CMP0048 NEW)

set(CMAKE_BUILD_TYPE_INIT "Release")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES native)

option(ENABLE_OMP_NVIDIA "Enable NVIDIA OpenMP build" OFF)
if (ENABLE_OMP_NVIDIA)
  option(FIND_PACKAGE_NVHPC "Use Find-Package for NVHPC detection" ON)
  if (FIND_PACKAGE_NVHPC)
    #  list(APPEND CMAKE_MODULE_PATH "/opt/nvidia/hpc_sdk/Linux_x86_64/2025/cmake/")
    list(APPEND CMAKE_PREFIX_PATH "/opt/nvidia/hpc_sdk/Linux_x86_64/2025/cmake/")
    find_package(NVHPC REQUIRED COMPONENTS CUDA MATH)
  else()
    set(CMAKE_CXX_COMPILER /opt/nvidia/hpc_sdk/Linux_x86_64/25.9/compilers/bin/nvc++)
    set(CMAKE_C_COMPILER /opt/nvidia/hpc_sdk/Linux_x86_64/25.9/compilers/bin/nvc)
    set(CMAKE_CUDA_COMPILER /opt/nvidia/hpc_sdk/Linux_x86_64/25.9/compilers/bin/nvcc)
    set(CMAKE_CXX_FLAGS -mp=gpu)
  endif()
endif()


project(DeePeeRT VERSION 0.1.0 LANGUAGES C CXX)

option(DEEPEE_ENABLE_CUDA "use CUDA" ON)
if (DEEPEE_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  enable_language(CUDA)
  set(DP_HAVE_CUDA ON)
endif()

if(NOT SET_UP_CONFIGURATIONS_DONE)
  set(SET_UP_CONFIGURATIONS_DONE 1)

  # No reason to set CMAKE_CONFIGURATION_TYPES if it's not a multiconfig generator
  # Also no reason mess with CMAKE_BUILD_TYPE if it's a multiconfig generator.
  if(CMAKE_CONFIGURATION_TYPES) # multiconfig generator?
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE) 
  else()
    if(NOT CMAKE_BUILD_TYPE)
      set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
    endif()
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY HELPSTRING "Choose the type of build")
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug;Release")
  endif()
endif()

if ((${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR}))
  # this is a local build, not a subproject
  SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
  SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
  SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
endif()




#find_package(OpenMP REQUIRED)

#CMAKE_CXX_COMPILER:STRING=/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/compilers/bin/nvc++
#CMAKE_C_COMPILER:STRING=/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/compilers/bin/nvc
#CMAKE_CUDA_COMPILER:STRING=/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/compilers/bin/nvcc
#CMAKE_CXX_FLAGS:STRING=-mp=gpu





option(DEEPEE_USE_LOCAL_CUBQL "Use *local* cuBQL" OFF)
mark_as_advanced(DEEPEE_USE_LOCAL_CUBQL)
if (DEEPEE_USE_LOCAL_CUBQL)
  add_subdirectory(../cuBQL EXCLUDE_FROM_ALL builddir_cuBQL)
else()
  add_subdirectory(submodules/cuBQL EXCLUDE_FROM_ALL builddir_cuBQL)
endif()

add_subdirectory(dp)

if (NVHPC_FOUND)
  target_link_libraries(deepeeRT PUBLIC
    NVHPC::CUDA NVHPC::MATH
  )
endif()

add_subdirectory(miniapp)


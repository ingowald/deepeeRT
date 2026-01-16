
option(DPC_CUDA "Use CUDA Backend?" OFF)
option(DPC_OMP  "Use OPENMP Backend?" OFF)
option(DPC_HIP  "Use HIP Backend?" OFF)

if (DPC_CUDA)
  add_subdirectory(cuda)
elseif (DPC_OMP)
  message("compiling with openmp backend")
  add_subdirectory(omp)
elseif (DPC_HIP)
  add_subdirectory(hip)
else()
  add_subdirectory(cuda)
endif()



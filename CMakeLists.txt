cmake_minimum_required (VERSION 2.6)
project (cudnn-training)

option(USE_CUDA "Build for CUDA platform" ON)
option(USE_OPENCL "Build for OpenCL. Needs CUDA-on-CL (beta)" OFF)

if(USE_CUDA)
find_package(CUDA 6.5 REQUIRED)

# Uncomment the following line to use gflags
#set(USE_GFLAGS 1)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Debug mode")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_50,code=compute_50;-std=c++11;-g;-lineinfo;-Xcompiler;-ggdb)
else()
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_50,code=compute_50;-std=c++11;-O3;-DNDEBUG;-Xcompiler;-DNDEBUG)
endif()

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Addresses a bug where code is not compiled as C++11 in non-CUDA code and older g++ versions
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if(USE_GFLAGS)
  add_definitions(-DUSE_GFLAGS)
endif()

include_directories($ENV{CUDNN_PATH} $ENV{CUDNN_PATH}/include)
link_directories($ENV{CUDNN_PATH} $ENV{CUDNN_PATH}/lib $ENV{CUDNN_PATH}/lib64)

cuda_add_executable(trainlenet lenet.cu readubyte.cpp)
cuda_add_cublas_to_target(trainlenet)

if(USE_GFLAGS)
  target_link_libraries(trainlenet gflags cudnn)
else()
  target_link_libraries(trainlenet cudnn)
endif()
else(USE_CUDA)
unset(CUDA_HOST_COMPILER CACHE)
unset(CUDA_TOOLKIT_ROOT_DIR CACHE)
unset(CUDA_SDK_ROOT_DIR CACHE)
unset(CUDA_USE_STATIC_CUDA_RUNTIME CACHE)
unset(CUDA_rt_LIBRARY CACHE)
endif(USE_CUDA)

if(USE_OPENCL)
option(INDEXES_32BIT "Needed on beignet 1.2.1, hd5500, I think" ON)
set(CLANG_HOME "/usr/lib/llvm-3.8" CACHE STRING "the clang-3.8.0 folder, containing lib, bin etc")
include(/usr/local/share/cocl/cocl.cmake)
add_cocl_executable(lenet lenet.cu readubyte.cpp)
else(USE_OPENCL)
unset(CLANG_HOME CACHE)
unset(INDEXES_32BIT CACHE)
endif()

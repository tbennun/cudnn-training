cmake_minimum_required (VERSION 2.6)
project (cudnn-training)

find_package(CUDA 6.5 REQUIRED)

# Uncomment the following line to use gflags
#set(USE_GFLAGS 1)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Debug mode")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_50,code=compute_50;-std=c++11;-g;-lineinfo;-Xcompiler;-ggdb)
else()
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_50,code=compute_50;-std=c++11;-O3;-DNDEBUG;-Xcompiler;-DNDEBUG)
endif()

list(APPEND CMAKE_CXX_FLAGS "-std=c++11")

if(USE_GFLAGS)
  add_definitions(-DUSE_GFLAGS)
endif()

include_directories($ENV{CUDNN_PATH})
link_directories($ENV{CUDNN_PATH})

cuda_add_executable(trainlenet lenet.cu readubyte.cpp)
cuda_add_cublas_to_target(trainlenet)

if(USE_GFLAGS)
  target_link_libraries(trainlenet gflags cudnn)
else()
  target_link_libraries(trainlenet cudnn)
endif()



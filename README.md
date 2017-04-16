A CUDNN minimal deep learning training code sample using LeNet.

Prerequisites
=============

* C++11 capable compiler (Visual Studio 2013, GCC 4.9 etc.) (for chrono and random)
* CUDA (6.5 or newer): https://developer.nvidia.com/cuda-downloads
* CUDNN (v5 or newer): https://developer.nvidia.com/cuDNN/
* MNIST dataset: http://yann.lecun.com/exdb/mnist/
* (optional) gflags: https://github.com/gflags/gflags

Set the CUDNN_PATH environment variable to where CUDNN is installed.


Compilation
===========

The project can either be compiled with CMake (cross-platform) or Visual Studio.

To compile with CMake, run the following commands:
```bash
~: $ cd cudnn-training/
~/cudnn-training: $ mkdir build
~/cudnn-training: $ cd build/
~/cudnn-training/build: $ cmake ..
~/cudnn-training/build: $ make
```

If compiling under linux, make sure to either set the ```CUDNN_PATH``` environment variable to the path CUDNN is installed to, or extract CUDNN to the CUDA toolkit path.

To enable gflags support, uncomment the line in CMakeLists.txt. In the Visual Studio project, define the macro ```USE_GFLAGS```.

Running
=======

Extract the MNIST training and test set files (*-ubyte) to a directory (if gflags are not used, the default is the current path).

You can also use the pre-trained weights published along with CUDNN, using the "pretrained" flag.

OpenCL
======

Building lenet using [cuda-on-cl](https://github.com/hughperkins/cuda-on-cl) is possible, although quite beta, and not very speedy. Please address any questions or issues to [https://github.com/hughperkins/cuda-on-cl/issues](cuda-on-cl issues).

To build for OpenCL:
- first, install cuda-on-cl, see https://github.com/hughperkins/cuda-on-cl#how-to-build
- in the cudnn cmake configuration screen:
  - set `USE_CUDA` to `OFF`, and `USE_CUDA` to `ON`
  - point `CLANG_HOME` at llvm-3.8 directory, eg `/usr/lib/llvm-3.8`, on Ubuntu 16.04

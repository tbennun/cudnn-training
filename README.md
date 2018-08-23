NOTE:
=====

- this project was forked from https://github.com/tbennun/cudnn-training
- additionally RElu activaitons for the biased convolution layers were applied
- based on http://cs231n.github.io/neural-networks-1/ the RElu activation is done after the bias was applied
- tested on Win10 64bit CUDA 9.0 VS2017 (toolset v140)  using cudnn64_7.dll, cublas64_90.dll, cudart64_90.dll
- the CMAKE file from forked repo does not work in VS2017 (since I don't use CMAKE, instead I created a new project in VS2017)
- added CompileCU.bat (to compile lenet.cu with nvcc.exe using VS2015 cl.exe and CUDA 9.0 toolkit to an .obj file, which then is linked   to the project)
- in the VS2017 project property settings  $(CUDA_PATH)\include needs to be added to the "Additional Include Directories"
- all these dependencies were added in in Linker/Additional Dependencies:
   cudnn\cudnn.lib
   cudnn\lenet.obj
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64\cuda.lib
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64\cudart_static.lib
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64\cublas.lib
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64\cufft.lib
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64\cufftw.lib
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64\curand.lib
- cudnn.h also must be accessable by the project   
- no pre-compiled headers
- MT static run-time library linking
- Optimization /O2 /Ot
- results after adding the RElu activations: 
   Training dataset size: 60000, Test dataset size: 10000 Batch size: 32, iterations: 200000
   Classification result: 0.91% error (used 10000 images)

---------------------------------------------------------------------------------------------------------------
Thank You tbennun for the Code! Great Project!

Info From Original Readme:




A CUDNN minimal deep learning training code sample using LeNet.

Prerequisites
=============

* C++11 capable compiler (Visual Studio 2013, GCC 4.9 etc.) (for chrono and random)
* CUDA (6.5 or newer): https://developer.nvidia.com/cuda-downloads
* CUDNN (v5, v6): https://developer.nvidia.com/cuDNN/
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

You can also load and save pre-trained weights (e.g., published along with CUDNN), using the "pretrained" and "save_data" flags respectively.

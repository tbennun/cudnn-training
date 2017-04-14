A CUDNN minimal deep learning training code sample using LeNet.

Original readme: https://github.com/tbennun/cudnn-training

Notes on this fork:
- work in progress to build using [cuda-on-cl](https://github.com/hughperkins/cuda-on-cl)
- what's working:
  - cublas calls: SGEMM, SGEMV
  - forward convolution, backward data, backward filter
- things that arent implemented yet:
  - activation forward/backward
  - pooling forward/backward
  - convolution bias backprop

## To build

- First, make sure you have installed the latest https://github.com/hughperkins/cuda-on-cl
- Then do:
```
git clone https://github.com/hughperkins/cudnn-training
cd cudnn-training
mkdir build
cd build
ccmake ..
# select 'opencl', deselect 'cuda'
# select path to llvm-3.8 (eg `/usr/lib/llvm-3.8` on ubuntu)
# set INDEXES_32BIT to 'ON'
# press 'c' then 'g'
make
./lenet
```

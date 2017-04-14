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

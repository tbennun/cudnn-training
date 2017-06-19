A CUDNN minimal deep learning training code sample using LeNet.

Original readme: https://github.com/tbennun/cudnn-training

This fork enables the cmake option `USE_OPENCL`, to build on OpenCL 1.2

## To build

- Install latest [coriander](https://github.com/hughperkins/coriander)
- Install the [coriander-dnn](https://github.com/hughperkins/coriander-dnn) plugin:
```
cocl_plugins.py install --repo-url https://github.com/hughperkins/coriander-dnn
```
- Then do:
```
git clone https://github.com/hughperkins/cudnn-training
cd cudnn-training
mkdir build
cd build
ccmake ..
# set `USE_CUDA` to `OFF`, and `USE_OPENCL` to `ON`
# press 'c' then 'g'
make
./lenet
```

Running
=======

Extract the MNIST training and test set files (*-ubyte) to a directory (if gflags are not used, the default is the current path).

You can also use the pre-trained weights published along with CUDNN, using the "pretrained" flag.

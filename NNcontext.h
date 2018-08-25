#pragma once

// Disclaimer: No Warranty. Use at your own risk.  (See License Information in Root)


#define CUDNN_MAJOR 7


#define TEST_ALPHA 1.0 
#define TEST_BETA   0.0 

///////////////////////////////////////////////////////////////////////////////////////////
// CUDNN/CUBLAS training context

struct TrainingContext
{
    cudnnHandle_t cudnnHandle = NULL;
    cublasHandle_t cublasHandle;

    cudnnTensorDescriptor_t dataTensor, conv1Tensor, conv1BiasTensor, pool1Tensor, 
                             conv2Tensor, conv2BiasTensor, pool2Tensor, fc1Tensor, fc2Tensor;
    cudnnFilterDescriptor_t conv1filterDesc, conv2filterDesc;
    cudnnConvolutionDescriptor_t conv1Desc, conv2Desc;
    cudnnConvolutionFwdAlgo_t conv1algo, conv2algo;
    cudnnConvolutionBwdFilterAlgo_t conv1bwfalgo, conv2bwfalgo;
    cudnnConvolutionBwdDataAlgo_t conv2bwdalgo;
    cudnnPoolingDescriptor_t poolDesc;
    cudnnActivationDescriptor_t fc1Activation;

    cudnnActivationDescriptor_t conv1Activation; 


    int m_gpuid;
    int m_batchSize;
    size_t m_workspaceSize;

    FullyConnectedLayer& ref_fc1, &ref_fc2;

    // Disable copying
    TrainingContext& operator=(const TrainingContext&) = delete;
    TrainingContext(const TrainingContext&) = delete;

    TrainingContext(int gpuid, int batch_size,
                    ConvBiasLayer& conv1, MaxPoolLayer& pool1, ConvBiasLayer& conv2, MaxPoolLayer& pool2,
                    FullyConnectedLayer& fc1, FullyConnectedLayer& fc2) : ref_fc1(fc1), ref_fc2(fc2), m_gpuid(gpuid)
    {
        m_batchSize = batch_size;

        b1_t = b1; 
        b2_t = b2; 
    

        // Create CUBLAS and CUDNN handles
        checkCudaErrors(cudaSetDevice(gpuid));
        checkCudaErrors(cublasCreate(&cublasHandle));
        checkCUDNN(cudnnCreate(&cudnnHandle));

        // Create tensor descriptors
        checkCUDNN(cudnnCreateTensorDescriptor(&dataTensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&conv1Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&conv1BiasTensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&pool1Tensor));

        checkCUDNN(cudnnCreateActivationDescriptor(&conv1Activation));

        checkCUDNN(cudnnCreateTensorDescriptor(&conv2Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&conv2BiasTensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&pool2Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&fc1Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&fc2Tensor));

        checkCUDNN(cudnnCreateActivationDescriptor(&fc1Activation));


        checkCUDNN(cudnnCreateFilterDescriptor(&conv1filterDesc));
        checkCUDNN(cudnnCreateFilterDescriptor(&conv2filterDesc));

        checkCUDNN(cudnnCreateConvolutionDescriptor(&conv1Desc));
        checkCUDNN(cudnnCreateConvolutionDescriptor(&conv2Desc));

        checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));            

        
        // Set tensor descriptor sizes
        checkCUDNN(cudnnSetTensor4dDescriptor(conv1BiasTensor,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              1, conv1.out_channels,
                                              1, 1));
        checkCUDNN(cudnnSetTensor4dDescriptor(conv2BiasTensor,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              1, conv2.out_channels,
                                              1, 1));
            
        checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc,
                                               CUDNN_POOLING_MAX,
                                               CUDNN_PROPAGATE_NAN,
                                               pool1.size, pool1.size,
                                               0, 0,
                                               pool1.stride, pool1.stride));
        checkCUDNN(cudnnSetTensor4dDescriptor(pool2Tensor,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              batch_size, conv2.out_channels,
                                              conv2.out_height / pool2.stride,
                                              conv2.out_width / pool2.stride));

        checkCUDNN(cudnnSetTensor4dDescriptor(fc1Tensor,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              batch_size, fc1.outputs, 1, 1));

        checkCUDNN(cudnnSetTensor4dDescriptor(fc2Tensor,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              batch_size, fc2.outputs, 1, 1));

        checkCUDNN(cudnnSetActivationDescriptor(fc1Activation, CUDNN_ACTIVATION_RELU,
                                                CUDNN_PROPAGATE_NAN, 0.0));


        checkCUDNN(cudnnSetActivationDescriptor(conv1Activation, CUDNN_ACTIVATION_RELU,   // M2018
                                                CUDNN_PROPAGATE_NAN, 0.0));

                

        // Set convolution tensor sizes and compute workspace size
        size_t workspace = 0;
        workspace = std::max(workspace, SetFwdConvolutionTensors(conv1, dataTensor, conv1Tensor, conv1filterDesc, conv1Desc, conv1algo));
        workspace = std::max(workspace, SetBwdConvolutionTensors(dataTensor, conv1Tensor, conv1filterDesc, conv1Desc, &conv1bwfalgo, nullptr));

        workspace = std::max(workspace, SetFwdConvolutionTensors(conv2, pool1Tensor, conv2Tensor, conv2filterDesc, conv2Desc, conv2algo));
        workspace = std::max(workspace, SetBwdConvolutionTensors(pool1Tensor, conv2Tensor, conv2filterDesc, conv2Desc, &conv2bwfalgo, &conv2bwdalgo));

        // The workspace is allocated later (if necessary)
        m_workspaceSize = workspace;
    }


    ~TrainingContext()
    {
        checkCudaErrors(cudaSetDevice(m_gpuid));

        checkCudaErrors(cublasDestroy(cublasHandle));
        checkCUDNN(cudnnDestroy(cudnnHandle));
        checkCUDNN(cudnnDestroyTensorDescriptor(dataTensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(conv1Tensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(conv1BiasTensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(pool1Tensor));

        checkCUDNN(cudnnDestroyActivationDescriptor(conv1Activation));

        checkCUDNN(cudnnDestroyTensorDescriptor(conv2Tensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(conv2BiasTensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(pool2Tensor));

        checkCUDNN(cudnnDestroyTensorDescriptor(fc1Tensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(fc2Tensor));
        checkCUDNN(cudnnDestroyActivationDescriptor(fc1Activation));
        checkCUDNN(cudnnDestroyFilterDescriptor(conv1filterDesc));
        checkCUDNN(cudnnDestroyFilterDescriptor(conv2filterDesc));
        checkCUDNN(cudnnDestroyConvolutionDescriptor(conv1Desc));
        checkCUDNN(cudnnDestroyConvolutionDescriptor(conv2Desc));
        checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));

        RemoveDropOut();
    }




 // taken from: https://devtalk.nvidia.com/default/topic/1028240/cudnn/how-to-implement-a-dropout-layer-using-cudnn-/
// unclassed to a pure function within this class:

    bool UseDropOut = false;

	  cudnnDropoutDescriptor_t dropout_descriptor = NULL;
	  size_t dropout_state_size;
	  size_t dropout_reserve_size;

    cudnnTensorDescriptor_t dropout_in_out_descriptor = NULL;

	  float dropRate;
	  float* ref_input{nullptr};
	  float* d_dropout_out{nullptr};
	  float* d_dx_dropout{nullptr};
	  void* states=NULL;
	  void* dropout_reserve_space=NULL;
	  int batchSize, features, imgH, imgW;
	  int in_out_bytes;


	int InitDropout(float dropRate, int batchSize, int features, int imageH, int imageW) 
	{
		in_out_bytes = sizeof(float)*batchSize*features*imageH*imageW;
		
    if (dropRate <= 0.0f) return 1; // avoid DropOut Layer
    if (!cudnnHandle) return -1; // failed


		checkCUDNN(cudnnCreateDropoutDescriptor(&dropout_descriptor));
		checkCUDNN(cudnnCreateTensorDescriptor(&dropout_in_out_descriptor));

		checkCUDNN(cudnnSetTensor4dDescriptor(dropout_in_out_descriptor,
											  CUDNN_TENSOR_NCHW,
											  CUDNN_DATA_FLOAT,
											  batchSize,
											  features,
											  imageH,
											  imageW));

		checkCUDNN(cudnnDropoutGetStatesSize(cudnnHandle, &dropout_state_size));

		checkCUDNN(cudnnDropoutGetReserveSpaceSize(dropout_in_out_descriptor, &dropout_reserve_size));

		// Allocate memory for states and reserve space
		checkCudaErrors(cudaMalloc(&states,dropout_state_size));
		checkCudaErrors(cudaMalloc(&dropout_reserve_space,dropout_reserve_size));

		checkCUDNN(cudnnSetDropoutDescriptor(dropout_descriptor,
											cudnnHandle,
											dropRate,
											states,
											dropout_state_size,
											/*Seed*/time(NULL)));

		checkCudaErrors(cudaMalloc(&d_dropout_out, in_out_bytes));
		checkCudaErrors(cudaMalloc(&d_dx_dropout, in_out_bytes));



    UseDropOut = true; // M2018

    return 0; // ok
	}



  void RemoveDropOut()
  {

    if (states)
    {
      checkCudaErrors(cudaFree(states));
    }
    if (dropout_reserve_space)
    {
      checkCudaErrors(cudaFree(dropout_reserve_space));
    }
    if (d_dropout_out)
    {
      checkCudaErrors(cudaFree(d_dropout_out));
    }
    if (d_dx_dropout)
    {
      checkCudaErrors(cudaFree(d_dx_dropout));
    }


    if (dropout_descriptor)
    {
      checkCUDNN(cudnnDestroyDropoutDescriptor(dropout_descriptor));
      dropout_descriptor = NULL;
    }

    if (dropout_in_out_descriptor)
    {
      checkCUDNN(cudnnDestroyTensorDescriptor(dropout_in_out_descriptor));
      dropout_in_out_descriptor = NULL;
    }
  }




    size_t SetFwdConvolutionTensors(ConvBiasLayer& conv, cudnnTensorDescriptor_t& srcTensorDesc, cudnnTensorDescriptor_t& dstTensorDesc,
                                    cudnnFilterDescriptor_t& filterDesc, cudnnConvolutionDescriptor_t& convDesc, 
                                    cudnnConvolutionFwdAlgo_t& algo)
    {
        size_t sizeInBytes = 0;

        int n = m_batchSize;
        int c = conv.in_channels;
        int h = conv.in_height;
        int w = conv.in_width;

        checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              n, c,
                                              h, w));

        checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,
                                              CUDNN_DATA_FLOAT,
                                              CUDNN_TENSOR_NCHW,
                                              conv.out_channels,
                                              conv.in_channels, 
                                              conv.kernel_size,
                                              conv.kernel_size));

//#if CUDNN_MAJOR > 5  // made for CUDNN 7
        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                                   0, 0,
                                                   1, 1,
                                                   1, 1,
                                                   CUDNN_CROSS_CORRELATION,
                                                   CUDNN_DATA_FLOAT));
//#else
//        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
//                                                   0, 0,
//                                                   1, 1,
//                                                   1, 1,
//                                                   CUDNN_CROSS_CORRELATION));
//#endif

        // Find dimension of convolution output
        checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc,
                                                         srcTensorDesc,
                                                         filterDesc,
                                                         &n, &c, &h, &w));

        checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              n, c,
                                              h, w));
        checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                                                       srcTensorDesc,
                                                       filterDesc,
                                                       convDesc,
                                                       dstTensorDesc,
                                                       CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                       0,
                                                       &algo));
        
        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                           srcTensorDesc,
                                                           filterDesc,
                                                           convDesc,
                                                           dstTensorDesc,
                                                           algo,
                                                           &sizeInBytes));

        return sizeInBytes;
    }




    void ForwardPropagation(float *data, float *conv1, float *conv1relu, float *pool1, 
                            float *conv2, float *conv2relu, float *pool2,  
                            float *fc1, float *fc1relu,
                            float *fc2, float *result,
                            float *pconv1, float *pconv1bias, 
                            float *pconv2, float *pconv2bias, 
                            float *pfc1, float *pfc1bias,
                            float *pfc2, float *pfc2bias, void *workspace, float *onevec)
    {        
        float alpha = 1.0f, beta = 0.0f;
        checkCudaErrors(cudaSetDevice(m_gpuid));

        // Conv1 layer
        checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, dataTensor,
                                           data, conv1filterDesc, pconv1, conv1Desc, 
                                           conv1algo, workspace, m_workspaceSize, &beta,
                                           conv1Tensor, conv1));
        checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, conv1BiasTensor,
                                  pconv1bias, &alpha, conv1Tensor, conv1));


        // http://cs231n.github.io/neural-networks-1/
        //  each neuron performs a dot product with the input and its weights, adds the bias 
        // and applies the non-linearity (or activation function)
        // => so do activation AFTER adding the BIAS

        // ReLU activation  
        float alphaCONV1 = TEST_ALPHA; 
        float betaCONV1 = TEST_BETA; 
        checkCUDNN(cudnnActivationForward(cudnnHandle, conv1Activation, &alphaCONV1,
                                          conv1Tensor, conv1, &betaCONV1, conv1Tensor, conv1relu));


        // Pool1 layer
        checkCUDNN(cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, conv1Tensor,
                                       conv1relu, &beta, pool1Tensor, pool1));   
        

        // Conv2 layer
        checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, pool1Tensor,
                                           pool1, conv2filterDesc, pconv2, conv2Desc,        
                                           conv2algo, workspace, m_workspaceSize, &beta,
                                           conv2Tensor, conv2));
        checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, conv2BiasTensor,
                                  pconv2bias, &alpha, conv2Tensor, conv2));


        // ReLU activation  
        float alphaCONV2 = TEST_ALPHA; 
        float betaCONV2 = TEST_BETA; 
        checkCUDNN(cudnnActivationForward(cudnnHandle, conv1Activation, &alphaCONV2,
                                          conv2Tensor, conv2, &betaCONV2, conv2Tensor, conv2relu));


        // Pool2 layer
        checkCUDNN(cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, conv2Tensor,
                                       conv2relu, &beta, pool2Tensor, pool2));   


        // FC1 layer
        // Forward propagate neurons using weights (fc1 = pfc1'*pool2)
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                    ref_fc1.outputs, m_batchSize, ref_fc1.inputs,
                                    &alpha,
                                    pfc1, ref_fc1.inputs,
                                    pool2, ref_fc1.inputs,                          
                                    &beta,
                                    fc1, ref_fc1.outputs));
        // Add bias using GEMM's "beta" (fc1 += pfc1bias*1_vec')
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    ref_fc1.outputs, m_batchSize, 1,
                                    &alpha,
                                    pfc1bias, ref_fc1.outputs,
                                    onevec, 1,
                                    &alpha,
                                    fc1, ref_fc1.outputs));

        // ReLU activation
        checkCUDNN(cudnnActivationForward(cudnnHandle, fc1Activation, &alpha,
                                          fc1Tensor, fc1, &beta, fc1Tensor, fc1relu));


        // on https://www.tensorflow.org/tutorials/estimators/cnn
        // dropout regularization is done after the RElu activation

        if (UseDropOut)
        {

          ref_input = fc1relu;

		       checkCUDNN(cudnnDropoutForward(cudnnHandle,
									   dropout_descriptor,
									   dropout_in_out_descriptor,
									   ref_input,
									   dropout_in_out_descriptor,
									   d_dropout_out,
									   dropout_reserve_space,
									   dropout_reserve_size));

           fc1relu = d_dropout_out;

        } // if (UseDropOut)



        // FC2 layer
        // Forward propagate neurons using weights (fc2 = pfc2'*fc1relu)
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                    ref_fc2.outputs, m_batchSize, ref_fc2.inputs,
                                    &alpha,
                                    pfc2, ref_fc2.inputs,
                                    fc1relu, ref_fc2.inputs,
                                    &beta,
                                    fc2, ref_fc2.outputs));
        // Add bias using GEMM's "beta" (fc2 += pfc2bias*1_vec')
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    ref_fc2.outputs, m_batchSize, 1,
                                    &alpha,
                                    pfc2bias, ref_fc2.outputs,
                                    onevec, 1,
                                    &alpha,
                                    fc2, ref_fc2.outputs));

        // Softmax loss
        checkCUDNN(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                       &alpha, fc2Tensor, fc2, &beta, fc2Tensor, result));
    }




    size_t SetBwdConvolutionTensors(cudnnTensorDescriptor_t& srcTensorDesc, cudnnTensorDescriptor_t& dstTensorDesc,
                                    cudnnFilterDescriptor_t& filterDesc, cudnnConvolutionDescriptor_t& convDesc, 
                                    cudnnConvolutionBwdFilterAlgo_t *falgo, cudnnConvolutionBwdDataAlgo_t *dalgo)
    {
        size_t sizeInBytes = 0, tmpsize = 0;

        // If backprop filter algorithm was requested
        if (falgo)
        {
            checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
                cudnnHandle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
                CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, falgo));

            checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                cudnnHandle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc, 
                *falgo, &tmpsize));

            sizeInBytes = std::max(sizeInBytes, tmpsize);
        }

        // If backprop data algorithm was requested
        if (dalgo)
        {
            checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
                cudnnHandle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc,
                CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, dalgo));

            checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
                cudnnHandle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc, 
                *dalgo, &tmpsize));

            sizeInBytes = std::max(sizeInBytes, tmpsize);
        }
        
        return sizeInBytes;
    }




    void Backpropagation(ConvBiasLayer& layer_conv1, MaxPoolLayer& layer_pool1, ConvBiasLayer& layer_conv2, MaxPoolLayer& layer_pool2,
                         float *data, float *labels, 
                         float *conv1, float *conv1relu,  float *pool1, 
                         float *conv2, float *conv2relu,  float *pool2,
                         float *fc1, float *fc1relu,
                         float *fc2, float *fc2smax, float *dloss_data,
                         float *pconv1, float *pconv1bias,
                         float *pconv2, float *pconv2bias,
                         float *pfc1, float *pfc1bias,
                         float *pfc2, float *pfc2bias,
                         float *gconv1, float *gconv1bias, float *dpool1,  float *dconv1relu,
                         float *gconv2, float *gconv2bias, float *dconv2, float *dpool2,  float *dconv2relu,
                         float *gfc1, float *gfc1bias, float *dfc1, float *dfc1relu,
                         float *gfc2, float *gfc2bias, float *dfc2,
                         void *workspace, float *onevec)
    {    
        float alpha = 1.0f, beta = 0.0f;

        float scalVal = 1.0f / static_cast<float>(m_batchSize);

        checkCudaErrors(cudaSetDevice(m_gpuid));

        // Initialization (using the training error function)
        checkCudaErrors(cudaMemcpyAsync(dloss_data, fc2smax, sizeof(float) * m_batchSize * ref_fc2.outputs, cudaMemcpyDeviceToDevice));
        
        // Softmax layer
        SoftmaxLossBackprop<<<RoundUp(m_batchSize, BW), BW>>>(labels, ref_fc2.outputs, m_batchSize, dloss_data);

        // Accounting for batch size in SGD
        checkCudaErrors(cublasSscal(cublasHandle, ref_fc2.outputs * m_batchSize, &scalVal, dloss_data, 1));

        // FC2 layer
        // Compute derivative with respect to weights: gfc2 = (fc1relu * dfc2smax')
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ref_fc2.inputs, ref_fc2.outputs, m_batchSize,
                                    &alpha, fc1relu, ref_fc2.inputs, dloss_data, ref_fc2.outputs, &beta, gfc2, ref_fc2.inputs));
        // Compute derivative with respect to bias: gfc2bias = dfc2smax * 1_vec
        checkCudaErrors(cublasSgemv(cublasHandle, CUBLAS_OP_N, ref_fc2.outputs, m_batchSize,
                                    &alpha, dloss_data, ref_fc2.outputs, onevec, 1, &beta, gfc2bias, 1));
        // Compute derivative with respect to data (for previous layer): pfc2*dfc2smax (500x10*10xN)
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ref_fc2.inputs, m_batchSize, ref_fc2.outputs,
                                    &alpha, pfc2, ref_fc2.inputs, dloss_data, ref_fc2.outputs, &beta, dfc2, ref_fc2.inputs));
        
        
        if (UseDropOut)
        {

          float* d_in_grads = dfc2;

      		checkCUDNN(cudnnDropoutBackward(cudnnHandle,
										dropout_descriptor,
										dropout_in_out_descriptor,
										d_in_grads,
										dropout_in_out_descriptor,
										d_dx_dropout,
										dropout_reserve_space,
										dropout_reserve_size));

           dfc2 = d_dx_dropout;

        } // if (UseDropOut)

        

        // ReLU activation
        checkCUDNN(cudnnActivationBackward(cudnnHandle, fc1Activation, &alpha,
                                           fc1Tensor, fc1relu, fc1Tensor, dfc2,
                                           fc1Tensor, fc1, &beta, fc1Tensor, dfc1relu));

        // FC1 layer
        // Compute derivative with respect to weights: gfc1 = (pool2 * dfc1relu')
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ref_fc1.inputs, ref_fc1.outputs, m_batchSize,
                                    &alpha, pool2, ref_fc1.inputs, dfc1relu, ref_fc1.outputs, &beta, gfc1, ref_fc1.inputs));
        // Compute derivative with respect to bias: gfc1bias = dfc1relu * 1_vec
        checkCudaErrors(cublasSgemv(cublasHandle, CUBLAS_OP_N, ref_fc1.outputs, m_batchSize,
                                    &alpha, dfc1relu, ref_fc1.outputs, onevec, 1, &beta, gfc1bias, 1));
        // Compute derivative with respect to data (for previous layer): pfc1*dfc1relu (800x500*500xN)
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ref_fc1.inputs, m_batchSize, ref_fc1.outputs,
                                    &alpha, pfc1, ref_fc1.inputs, dfc1relu, ref_fc1.outputs, &beta, dfc1, ref_fc1.inputs));



        // Pool2 layer
        checkCUDNN(cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha, 
                                        pool2Tensor, pool2, pool2Tensor, dfc1,  
                                        conv2Tensor, conv2, &beta, conv2Tensor, dpool2)); 


        // ReLU activation 
        float alphaCONV2 = TEST_ALPHA; 
        float betaCONV2 = TEST_BETA; 
        checkCUDNN(cudnnActivationBackward(cudnnHandle, conv1Activation, &alphaCONV2,  // re-use conv1Activation also for conv2
                                           conv2Tensor, conv2relu, conv2Tensor, dpool2,   
                                           conv2Tensor, conv2, &betaCONV2, conv2Tensor, dconv2relu));


        // Conv2 layer
        checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle, &alpha, conv2Tensor,
                                                dconv2relu, &beta, conv2BiasTensor, gconv2bias));     

        
        checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, pool1Tensor,
                                                  pool1, conv2Tensor, dpool2, conv2Desc,
                                                  conv2bwfalgo, workspace, m_workspaceSize,
                                                  &beta, conv2filterDesc, gconv2));
    
        checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle, &alpha, conv2filterDesc,
                                                pconv2, conv2Tensor, dpool2, conv2Desc, 
                                                conv2bwdalgo, workspace, m_workspaceSize,
                                                &beta, pool1Tensor, dconv2));
    


        // Pool1 layer
        checkCUDNN(cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha,
                                          pool1Tensor, pool1, pool1Tensor, dconv2,                   
                                          conv1Tensor, conv1, &beta, conv1Tensor, dpool1));


        // ReLU activation
        float alphaCONV1 = TEST_ALPHA; 
        float betaCONV1 = TEST_BETA; 
        checkCUDNN(cudnnActivationBackward(cudnnHandle, conv1Activation, &alphaCONV1,
                                           conv1Tensor, conv1relu, conv1Tensor, dpool1,   
                                           conv1Tensor, conv1, &betaCONV1, conv1Tensor, dconv1relu));


        // Conv1 layer
        checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle, &alpha, conv1Tensor,
                                                dconv1relu, &beta, conv1BiasTensor, gconv1bias));  
        
        checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, dataTensor,
                                                  data, conv1Tensor, dpool1, conv1Desc,
                                                  conv1bwfalgo, workspace, m_workspaceSize,
                                                  &beta, conv1filterDesc, gconv1));

        // No need for convBackwardData because there are no more layers below
    }

    // Adam
    const float b1 = Beta1; // 0.9f;     // decay term
    const float b2 = Beta2; //0.999f; // decay term
    float b1_t = Beta1; //0.9f;        // decay term power t
    float b2_t = Beta2; //0.999f;    // decay term power t
    
 


    void UpdateWeights(float learning_rate, float momentum,
                       ConvBiasLayer& conv1, ConvBiasLayer& conv2,
                       float *pconv1, float *pconv1bias,
                       float *pconv2, float *pconv2bias,
                       float *pfc1, float *pfc1bias,
                       float *pfc2, float *pfc2bias,
                       float *gconv1, float *gconv1bias,
                       float *gconv2, float *gconv2bias,
                       float *gfc1, float *gfc1bias,
                       float *gfc2, float *gfc2bias,

                       float *vconv1,  float *vconv1bias,
                       float *vconv2,   float *vconv2bias,
                       float *vfc1,        float *vfc1bias,
                       float *vfc2,        float *vfc2bias,

                       float *mconv1,  float *mconv1bias,
                       float *mconv2,   float *mconv2bias,
                       float *mfc1,        float *mfc1bias,
                       float *mfc2,        float *mfc2bias )

    {    
        float alpha = -learning_rate;

        checkCudaErrors(cudaSetDevice(m_gpuid));

       const int size_conv1 = static_cast<int>(conv1.pneurons.size());
       const int size_conv1bias = static_cast<int>(conv1.pbias.size());

       const int size_conv2 = static_cast<int>(conv2.pneurons.size());
       const int size_conv2bias = static_cast<int>(conv2.pbias.size());

       const int size_fc1 = static_cast<int>(ref_fc1.pneurons.size());
       const int size_fc1bias = static_cast<int>(ref_fc1.pbias.size());

       const int size_fc2 = static_cast<int>(ref_fc2.pneurons.size());
       const int size_fc2bias = static_cast<int>(ref_fc2.pbias.size());


       if (Nadamax)
       {
         float MomentumUpdate = momentum;
         float NextMomentumUpdate = MomentumUpdate;  // constant momentum (so simply same)

         // Conv1
         NadamaxWeightUpdate << <RoundUp(size_conv1, BW), BW >> > (pconv1, gconv1, vconv1, mconv1, alpha, MomentumUpdate, NextMomentumUpdate, b2, b1_t, b2_t, size_conv1);
         NadamaxWeightUpdate << <RoundUp(size_conv1bias, BW), BW >> > (pconv1bias, gconv1bias, vconv1bias, mconv1bias, alpha, MomentumUpdate, NextMomentumUpdate, b2, b1_t, b2_t, size_conv1bias);

         // Conv2
         NadamaxWeightUpdate << <RoundUp(size_conv2, BW), BW >> > (pconv2, gconv2, vconv2, mconv2, alpha, MomentumUpdate, NextMomentumUpdate, b2, b1_t,  b2_t, size_conv2);
         NadamaxWeightUpdate << <RoundUp(size_conv2bias, BW), BW >> > (pconv2bias, gconv2bias, vconv2bias, mconv2bias, alpha, MomentumUpdate, NextMomentumUpdate, b2, b1_t, b2_t,  size_conv2bias);

         // Fully connected 1
         NadamaxWeightUpdate << <RoundUp(size_fc1, BW), BW >> > (pfc1, gfc1, vfc1, mfc1, alpha, MomentumUpdate, NextMomentumUpdate, b2, b1_t, b2_t,  size_fc1);
         NadamaxWeightUpdate << <RoundUp(size_fc1bias, BW), BW >> > (pfc1bias, gfc1bias, vfc1bias, mfc1bias, alpha, MomentumUpdate, NextMomentumUpdate, b2, b1_t, b2_t, size_fc1bias);

         // Fully connected 2
         NadamaxWeightUpdate << <RoundUp(size_fc2, BW), BW >> > (pfc2, gfc2, vfc2, mfc2, alpha, MomentumUpdate, NextMomentumUpdate, b2, b1_t, b2_t,  size_fc2);
         NadamaxWeightUpdate << <RoundUp(size_fc2bias, BW), BW >> > (pfc2bias, gfc2bias, vfc2bias, mfc2bias, alpha, MomentumUpdate, NextMomentumUpdate, b2, b1_t, b2_t, size_fc2bias);

                  
         b1_t *= MomentumUpdate;
         b2_t *= b2;
         return;
       }
       else if (Nadam)
       {
         float MomentumUpdate = momentum;
         float NextMomentumUpdate = MomentumUpdate;  // constant momentum (so simply same)

         // Conv1
         NadamWeightUpdate << <RoundUp(size_conv1, BW), BW >> > (pconv1, gconv1, vconv1, mconv1, alpha, MomentumUpdate, NextMomentumUpdate, b2, b1_t, b2_t, size_conv1);
         NadamWeightUpdate << <RoundUp(size_conv1bias, BW), BW >> > (pconv1bias, gconv1bias, vconv1bias, mconv1bias, alpha, MomentumUpdate, NextMomentumUpdate, b2, b1_t, b2_t, size_conv1bias);

         // Conv2
         NadamWeightUpdate << <RoundUp(size_conv2, BW), BW >> > (pconv2, gconv2, vconv2, mconv2, alpha, MomentumUpdate, NextMomentumUpdate, b2, b1_t,  b2_t, size_conv2);
         NadamWeightUpdate << <RoundUp(size_conv2bias, BW), BW >> > (pconv2bias, gconv2bias, vconv2bias, mconv2bias, alpha, MomentumUpdate, NextMomentumUpdate, b2, b1_t, b2_t,  size_conv2bias);

         // Fully connected 1
         NadamWeightUpdate << <RoundUp(size_fc1, BW), BW >> > (pfc1, gfc1, vfc1, mfc1, alpha, MomentumUpdate, NextMomentumUpdate, b2, b1_t, b2_t,  size_fc1);
         NadamWeightUpdate << <RoundUp(size_fc1bias, BW), BW >> > (pfc1bias, gfc1bias, vfc1bias, mfc1bias, alpha, MomentumUpdate, NextMomentumUpdate, b2, b1_t, b2_t, size_fc1bias);

         // Fully connected 2
         NadamWeightUpdate << <RoundUp(size_fc2, BW), BW >> > (pfc2, gfc2, vfc2, mfc2, alpha, MomentumUpdate, NextMomentumUpdate, b2, b1_t, b2_t,  size_fc2);
         NadamWeightUpdate << <RoundUp(size_fc2bias, BW), BW >> > (pfc2bias, gfc2bias, vfc2bias, mfc2bias, alpha, MomentumUpdate, NextMomentumUpdate, b2, b1_t, b2_t, size_fc2bias);

                  
         b1_t *= MomentumUpdate;
         b2_t *= b2;
         return;
       }
       else if (Adamax)
       {

         // Conv1
         AdamaxWeightUpdate << <RoundUp(size_conv1, BW), BW >> > (pconv1, gconv1, vconv1, mconv1, alpha, b1, b2, b1_t, size_conv1);
         AdamaxWeightUpdate << <RoundUp(size_conv1bias, BW), BW >> > (pconv1bias, gconv1bias, vconv1bias, mconv1bias, alpha, b1, b2, b1_t, size_conv1bias);

         // Conv2
         AdamaxWeightUpdate << <RoundUp(size_conv2, BW), BW >> > (pconv2, gconv2, vconv2, mconv2, alpha, b1, b2, b1_t,  size_conv2);
         AdamaxWeightUpdate << <RoundUp(size_conv2bias, BW), BW >> > (pconv2bias, gconv2bias, vconv2bias, mconv2bias, alpha, b1, b2, b1_t,  size_conv2bias);

         // Fully connected 1
         AdamaxWeightUpdate << <RoundUp(size_fc1, BW), BW >> > (pfc1, gfc1, vfc1, mfc1, alpha, b1, b2, b1_t,  size_fc1);
         AdamaxWeightUpdate << <RoundUp(size_fc1bias, BW), BW >> > (pfc1bias, gfc1bias, vfc1bias, mfc1bias, alpha, b1, b2, b1_t, size_fc1bias);

         // Fully connected 2
         AdamaxWeightUpdate << <RoundUp(size_fc2, BW), BW >> > (pfc2, gfc2, vfc2, mfc2, alpha, b1, b2, b1_t, size_fc2);
         AdamaxWeightUpdate << <RoundUp(size_fc2bias, BW), BW >> > (pfc2bias, gfc2bias, vfc2bias, mfc2bias, alpha, b1, b2, b1_t, size_fc2bias);

                  
         b1_t *= b1;

         return;
       }
       else if (Adam)
       {


         // Conv1
         AdamWeightUpdate << <RoundUp(size_conv1, BW), BW >> > (pconv1, gconv1, vconv1, mconv1, alpha, b1, b2, b1_t, b2_t, size_conv1);
         AdamWeightUpdate << <RoundUp(size_conv1bias, BW), BW >> > (pconv1bias, gconv1bias, vconv1bias, mconv1bias, alpha, b1, b2, b1_t, b2_t, size_conv1bias);

         // Conv2
         AdamWeightUpdate << <RoundUp(size_conv2, BW), BW >> > (pconv2, gconv2, vconv2, mconv2, alpha, b1, b2, b1_t, b2_t, size_conv2);
         AdamWeightUpdate << <RoundUp(size_conv2bias, BW), BW >> > (pconv2bias, gconv2bias, vconv2bias, mconv2bias, alpha, b1, b2, b1_t, b2_t, size_conv2bias);

         // Fully connected 1
         AdamWeightUpdate << <RoundUp(size_fc1, BW), BW >> > (pfc1, gfc1, vfc1, mfc1, alpha, b1, b2, b1_t, b2_t, size_fc1);
         AdamWeightUpdate << <RoundUp(size_fc1bias, BW), BW >> > (pfc1bias, gfc1bias, vfc1bias, mfc1bias, alpha, b1, b2, b1_t, b2_t, size_fc1bias);

         // Fully connected 2
         AdamWeightUpdate << <RoundUp(size_fc2, BW), BW >> > (pfc2, gfc2, vfc2, mfc2, alpha, b1, b2, b1_t, b2_t, size_fc2);
         AdamWeightUpdate << <RoundUp(size_fc2bias, BW), BW >> > (pfc2bias, gfc2bias, vfc2bias, mfc2bias, alpha, b1, b2, b1_t, b2_t, size_fc2bias);


                  
         b1_t *= b1;
         b2_t *= b2;

         return;
       }



       if (FLAGS_momentum > 0.0)
       {
         

#if 1
         // Conv1
         NesterovMomentumWeightUpdate << <RoundUp(size_conv1, BW), BW >> > (pconv1, gconv1, vconv1, alpha, momentum, size_conv1);
         NesterovMomentumWeightUpdate << <RoundUp(size_conv1bias, BW), BW >> > (pconv1bias, gconv1bias, vconv1bias, alpha, momentum, size_conv1bias);

         // Conv2
         NesterovMomentumWeightUpdate << <RoundUp(size_conv2, BW), BW >> > (pconv2, gconv2, vconv2, alpha, momentum, size_conv2);
         NesterovMomentumWeightUpdate << <RoundUp(size_conv2bias, BW), BW >> > (pconv2bias, gconv2bias, vconv2bias, alpha, momentum, size_conv2bias);

         // Fully connected 1
         NesterovMomentumWeightUpdate << <RoundUp(size_fc1, BW), BW >> > (pfc1, gfc1, vfc1, alpha, momentum, size_fc1);
         NesterovMomentumWeightUpdate << <RoundUp(size_fc1bias, BW), BW >> > (pfc1bias, gfc1bias, vfc1bias, alpha, momentum, size_fc1bias);

         // Fully connected 2
         NesterovMomentumWeightUpdate << <RoundUp(size_fc2, BW), BW >> > (pfc2, gfc2, vfc2, alpha, momentum, size_fc2);
         NesterovMomentumWeightUpdate << <RoundUp(size_fc2bias, BW), BW >> > (pfc2bias, gfc2bias, vfc2bias, alpha, momentum, size_fc2bias);

         return;
#endif


#if  0 // SGD momentum
         // Momentum update    v=momentum=0.9
         // v[i] = mu * v[i] - learning_rate * dx # integrate velocity
         // x += v # integrate position
         //  const float MomentumUpdate = 0.9f;
         //  v_update[i] = v_update * MomentumUpdate - learning_rate;


          //SGDmomentumWeightUpdate(float *weights,  float *gradients, float *v, float learning_rate,  int size)

         // Conv1
         SGDmomentumWeightUpdate << <RoundUp(size_conv1, BW), BW >> > (pconv1, gconv1, vconv1, alpha, momentum, size_conv1);
         SGDmomentumWeightUpdate << <RoundUp(size_conv1bias, BW), BW >> > (pconv1bias, gconv1bias, vconv1bias, alpha, momentum, size_conv1bias);

         // Conv2
         SGDmomentumWeightUpdate << <RoundUp(size_conv2, BW), BW >> > (pconv2, gconv2, vconv2, alpha, momentum, size_conv2);
         SGDmomentumWeightUpdate << <RoundUp(size_conv2bias, BW), BW >> > (pconv2bias, gconv2bias, vconv2bias, alpha, momentum, size_conv2bias);

         // Fully connected 1
         SGDmomentumWeightUpdate << <RoundUp(size_fc1, BW), BW >> > (pfc1, gfc1, vfc1, alpha, momentum, size_fc1);
         SGDmomentumWeightUpdate << <RoundUp(size_fc1bias, BW), BW >> > (pfc1bias, gfc1bias, vfc1bias, alpha, momentum, size_fc1bias);

         // Fully connected 2
         SGDmomentumWeightUpdate << <RoundUp(size_fc2, BW), BW >> > (pfc2, gfc2, vfc2, alpha, size_fc2);
         SGDmomentumWeightUpdate << <RoundUp(size_fc2bias, BW), BW >> > (pfc2bias, gfc2bias, vfc2bias, alpha, momentum, size_fc2bias);


         return;
#endif
       }
       else
       {
#if 1 // SGD
         // Conv1
         checkCudaErrors(cublasSaxpy(cublasHandle, size_conv1, &alpha, gconv1, 1, pconv1, 1));
         checkCudaErrors(cublasSaxpy(cublasHandle, size_conv1bias, &alpha, gconv1bias, 1, pconv1bias, 1));

         // Conv2
         checkCudaErrors(cublasSaxpy(cublasHandle, size_conv2, &alpha, gconv2, 1, pconv2, 1));
         checkCudaErrors(cublasSaxpy(cublasHandle, size_conv2bias, &alpha, gconv2bias, 1, pconv2bias, 1));

         // Fully connected 1
         checkCudaErrors(cublasSaxpy(cublasHandle, size_fc1, &alpha, gfc1, 1, pfc1, 1));
         checkCudaErrors(cublasSaxpy(cublasHandle, size_fc1bias, &alpha, gfc1bias, 1, pfc1bias, 1));

         // Fully connected 2
         checkCudaErrors(cublasSaxpy(cublasHandle, size_fc2, &alpha, gfc2, 1, pfc2, 1));
         checkCudaErrors(cublasSaxpy(cublasHandle, size_fc2bias, &alpha, gfc2bias, 1, pfc2bias, 1));
#endif
       }
    }
};



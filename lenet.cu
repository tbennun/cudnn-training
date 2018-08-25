
// Disclaimer: No Warranty. Use at your own risk.  (See License Information in Root)
// this project was forked from https://github.com/tbennun/cudnn-training

#define CUDNN_MAJOR 7    // CUDA 9.0  cudnn64_7.dll

// choose zero or one from the next two lines   (none= Standard SGD)
//#define USE_NESTEROV_MOMENTUM
#define USE_ADAM

// comment the next line to use without dropout layer
//#define USE_DROPOUT_LAYER

// comment the next line to use default learning rate update (decaying ~1/T)
#define USE_SCHEDULED_LEARNING_RATE



// =========================================================================================================
// SWITCHES:


//#define ALLOW_SAVE_AS_PGM_FILE


// =========================================================================================================
// Definitions 


#define BATCH_SIZE  64 // 32 // 256 // 4096  // ORG=64
// BW = Block width for CUDA kernels
#define defaultBW  (BATCH_SIZE  * 2)      // 128
#define ITERATIONS   1000
#define LEARNING_RATE 0.001 //  ORG=0.01
#define LEARNING_RATE_POLICY_GAMMA 0.0001 // 0.00001    // 0.000001 // ORG: 0.0001
#define LEARNING_RATE_POLICY_POWER   0.85 // 0.8 // 0.31 // 0.51 // 0.2 // 0.69  // 0.75  // ORG: 0.75
// Adam:
#define BETA1 0.9f     // ORG: 0.9f
#define BETA2 0.999f // ORG:  0.999f


#define DROP_RATE 0.0 // 0.001 // 0.4      0.0=OFF
#define MOMENTUM 0.9  // 0.0=OFF



    // Constant versions of gflags (from https://github.com/tbennun/cudnn-training)
    #define DEFINE_int32(flag, default_value, description) const int FLAGS_##flag = (default_value)
    #define DEFINE_uint64(flag, default_value, description) const unsigned long long FLAGS_##flag = (default_value)
    #define DEFINE_bool(flag, default_value, description) const bool FLAGS_##flag = (default_value)
    #define DEFINE_double(flag, default_value, description) const double FLAGS_##flag = (default_value)
    #define DEFINE_string(flag, default_value, description) const std::string FLAGS_##flag ((default_value))







///////////////////////////////////////////////////////////////////////////////////////////
// Command-line flags


// Application parameters
DEFINE_int32(gpu, 0, "The GPU ID to use");
//DEFINE_int32(iterations, 1000, "Number of iterations for training"); // 7.8% Error rate (on classification)
//DEFINE_int32(iterations, ITERATIONS, "Number of iterations for training");
DEFINE_int32(random_seed, -1, "Override random seed (default uses std::random_device)");
DEFINE_int32(classify, -1, "Number of images to classify to compute error rate (default uses entire test set)");

unsigned long long FLAGS_batch_size = BATCH_SIZE;
unsigned long long BW = defaultBW; // BATCH_SIZE * 2;
int FLAGS_iterations = ITERATIONS;
double FLAGS_learning_rate = LEARNING_RATE; 
double FLAGS_lr_gamma = LEARNING_RATE_POLICY_GAMMA;
double FLAGS_lr_power = LEARNING_RATE_POLICY_POWER;
double FLAGS_momentum = MOMENTUM;
double FLAGS_drop_rate = DROP_RATE;

bool Adam = false;
float Beta1 = BETA1; //  0.9f;     // decay term
float Beta2 = BETA2; // 0.999f; // decay term

bool Adamax = false;
bool Nadam = false;
bool Nadamax = false;

bool ConstantLearningRate = false;


double ExponentialDecayK = 0.001f; 

double StepDecayScheduleDrop = 0.5; //  0.0=OFF
double StepDecayScheduleEpochsDrop = 250.0f;

DEFINE_bool(pretrained, false, "Use the pretrained CUDNN model as input");
DEFINE_bool(save_data, false, "Save pretrained weights to file");



///////////////////////////////////////////////////////////////////////////////////////////
// helper utilities

#include "helpers.h"
#include "readubyte.h"



// Filenames
DEFINE_string(train_images, "train-images.idx3-ubyte", "Training images filename");
DEFINE_string(train_labels, "train-labels.idx1-ubyte", "Training labels filename");
DEFINE_string(test_images, "t10k-images.idx3-ubyte", "Test images filename");
DEFINE_string(test_labels, "t10k-labels.idx1-ubyte", "Test labels filename");


///////////////////////////////////////////////////////////////////////////////////////////
// Layer representations

#include "NNlayers.h"


///////////////////////////////////////////////////////////////////////////////////////////
// GPU Kernels

#include "NNkernels.h"



///////////////////////////////////////////////////////////////////////////////////////////
// CUDNN/CUBLAS training context


#include "NNcontext.h"

n/a

///////////////////////////////////////////////////////////////////////////////////////////
// CUDNN/CUBLAS training context

struct TrainingContext
{
    cudnnHandle_t cudnnHandle;
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

    cudnnActivationDescriptor_t conv1Activation;  // also RElu  (here also "fc1Activation" could be re-used, if both always remain RElu's)
    
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

     #ifdef USE_ADAM
        b1_t = b1; 
        b2_t = b2; 	    
     #endif	    
	    
        // Create CUBLAS and CUDNN handles
        checkCudaErrors(cudaSetDevice(gpuid));
        checkCudaErrors(cublasCreate(&cublasHandle));
        checkCUDNN(cudnnCreate(&cudnnHandle));

        // Create tensor descriptors
        checkCUDNN(cudnnCreateTensorDescriptor(&dataTensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&conv1Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&conv1BiasTensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&pool1Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&conv2Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&conv2BiasTensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&pool2Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&fc1Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&fc2Tensor));

        checkCUDNN(cudnnCreateActivationDescriptor(&fc1Activation));
        checkCUDNN(cudnnCreateActivationDescriptor(&conv1Activation));

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

        checkCUDNN(cudnnSetActivationDescriptor(fc1Activation, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
        checkCUDNN(cudnnSetActivationDescriptor(conv1Activation, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));        
        
        
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
        checkCUDNN(cudnnDestroyTensorDescriptor(conv2Tensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(conv2BiasTensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(pool2Tensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(fc1Tensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(fc2Tensor));
        checkCUDNN(cudnnDestroyActivationDescriptor(fc1Activation));
        checkCUDNN(cudnnDestroyActivationDescriptor(conv1Activation));        
        checkCUDNN(cudnnDestroyFilterDescriptor(conv1filterDesc));
        checkCUDNN(cudnnDestroyFilterDescriptor(conv2filterDesc));
        checkCUDNN(cudnnDestroyConvolutionDescriptor(conv1Desc));
        checkCUDNN(cudnnDestroyConvolutionDescriptor(conv2Desc));
        checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
        #ifdef DROPOUT_LAYER
          RemoveDropOut();
        #endif
    }
    
    
    
#ifdef USE_DROPOUT_LAYER
  // taken from: https://devtalk.nvidia.com/default/topic/1028240/cudnn/how-to-implement-a-dropout-layer-using-cudnn-/
  // unclassed to pure functions within this class:

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
		
        if (dropRate <= 0.0f) return 1; // cancelled; avoid DropOut Layer
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



    	    UseDropOut = true; 

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
#endif

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

#if CUDNN_MAJOR > 5
        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                                   0, 0,
                                                   1, 1,
                                                   1, 1,
                                                   CUDNN_CROSS_CORRELATION,
                                                   CUDNN_DATA_FLOAT));
#else
        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                                   0, 0,
                                                   1, 1,
                                                   1, 1,
                                                   CUDNN_CROSS_CORRELATION));
#endif

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
        //  each neuron performs a dot product with the input and its weights (here convolution), adds the bias 
        // and applies the non-linearity (or activation function)
        // => so do activation AFTER adding the BIAS


        // ReLU activation
        float alphaCONV1 = 1.0f; 
        float betaCONV1 = 0.0f; 
        checkCUDNN(cudnnActivationForward(cudnnHandle, conv1Activation, &alphaCONV1,
                                          conv1Tensor, conv1, &betaCONV1, conv1Tensor, conv1relu));

        // Pool1 layer 
        checkCUDNN(cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, conv1Tensor,
                                       conv1relu, &beta, pool1Tensor, pool1));   // changed: conv1 to conv1relu            

        // Conv2 layer
        checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, pool1Tensor,
                                           pool1, conv2filterDesc, pconv2, conv2Desc, 
                                           conv2algo, workspace, m_workspaceSize, &beta,
                                           conv2Tensor, conv2));
        checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, conv2BiasTensor,
                                  pconv2bias, &alpha, conv2Tensor, conv2));
        
        
        // ReLU activation
        float alphaCONV2 = 1.0f; 
        float betaCONV2 = 0.0f; 
        checkCUDNN(cudnnActivationForward(cudnnHandle, conv1Activation, &alphaCONV2,
                                          conv2Tensor, conv2, &betaCONV2, conv2Tensor, conv2relu));

        // Pool2 layer
        checkCUDNN(cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, conv2Tensor,
                                       conv2relu, &beta, pool2Tensor, pool2));   // changed: conv2 to conv2relu
                            

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

        
#ifdef USE_DROPOUT_LAYER
        if (UseDropOut)
        {
          // on https://www.tensorflow.org/tutorials/estimators/cnn
          // dropout regularization is done after the RElu activatio
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
        }
#endif        
        
        
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
                         float *gconv1, float *gconv1bias, float *dpool1, float *dconv1relu,
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
        
        
#ifdef USE_DROPOUT_LAYER
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
        }
#endif
        
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
        float alphaCONV2 = 1.0f; 
        float betaCONV2 = 0.0f; 
        checkCUDNN(cudnnActivationBackward(cudnnHandle, conv1Activation, &alphaCONV2,  // re-use conv1Activation also for conv2
                                           conv2Tensor, conv2relu, conv2Tensor, dpool2,   
                                           conv2Tensor, conv2, &betaCONV2, conv2Tensor, dconv2relu));

        // Conv2 layer
        checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle, &alpha, conv2Tensor,
                                                dconv2relu, &beta, conv2BiasTensor, gconv2bias));     // changed:  dpool2 to dconv2relu
                
        
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
        float alphaCONV1 = 1.0f; 
        float betaCONV1 = 0.0f; 
        checkCUDNN(cudnnActivationBackward(cudnnHandle, conv1Activation, &alphaCONV1,
                                           conv1Tensor, conv1relu, conv1Tensor, dpool1,   
                                           conv1Tensor, conv1, &betaCONV1, conv1Tensor, dconv1relu));

        // Conv1 layer 
        checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle, &alpha, conv1Tensor,
                                                dconv1relu, &beta, conv1BiasTensor, gconv1bias));  // changed: dpool1 to dconv1relu
        
        checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, dataTensor,
                                                  data, conv1Tensor, dpool1, conv1Desc,
                                                  conv1bwfalgo, workspace, m_workspaceSize,
                                                  &beta, conv1filterDesc, gconv1));

        // No need for convBackwardData because there are no more layers below
    }

	
	
#ifdef USE_ADAM
    // Adam
    const float b1 = 0.9f;     // decay term
    const float b2 = 0.999f; // decay term
    float b1_t = 0.9f;        // decay term power t
    float b2_t = 0.999f;    // decay term power t	
#endif	
	
	
    void UpdateWeights(float learning_rate,
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
                       float *mfc2,        float *mfc2bias)
    {    
        float alpha = -learning_rate;
        
#ifdef USE_ADAM

       const int size_conv1 = static_cast<int>(conv1.pconv.size());
       const int size_conv1bias = static_cast<int>(conv1.pbias.size());

       const int size_conv2 = static_cast<int>(conv2.pconv.size());
       const int size_conv2bias = static_cast<int>(conv2.pbias.size());

       const int size_fc1 = static_cast<int>(ref_fc1.pneurons.size());
       const int size_fc1bias = static_cast<int>(ref_fc1.pbias.size());

       const int size_fc2 = static_cast<int>(ref_fc2.pneurons.size());
       const int size_fc2bias = static_cast<int>(ref_fc2.pbias.size());
	    
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
	    
#else	    
 #ifdef USE_NESTEROV_MOMENTUM
        
       const float momentum = 0.9f;
        
       const int size_conv1 = static_cast<int>(conv1.pconv.size());
       const int size_conv1bias = static_cast<int>(conv1.pbias.size());

       const int size_conv2 = static_cast<int>(conv2.pconv.size());
       const int size_conv2bias = static_cast<int>(conv2.pbias.size());

       const int size_fc1 = static_cast<int>(ref_fc1.pneurons.size());
       const int size_fc1bias = static_cast<int>(ref_fc1.pbias.size());

       const int size_fc2 = static_cast<int>(ref_fc2.pneurons.size());
       const int size_fc2bias = static_cast<int>(ref_fc2.pbias.size());
        
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
        
 #else        
        checkCudaErrors(cudaSetDevice(m_gpuid));

        // Conv1
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv1.pconv.size()),
                                    &alpha, gconv1, 1, pconv1, 1));
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv1.pbias.size()),
                                    &alpha, gconv1bias, 1, pconv1bias, 1));

        // Conv2
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv2.pconv.size()),
                                    &alpha, gconv2, 1, pconv2, 1));
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv2.pbias.size()),
                                    &alpha, gconv2bias, 1, pconv2bias, 1));

        // Fully connected 1
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.pneurons.size()),
                                    &alpha, gfc1, 1, pfc1, 1));
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.pbias.size()),
                                    &alpha, gfc1bias, 1, pfc1bias, 1));

        // Fully connected 2
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc2.pneurons.size()),
                                    &alpha, gfc2, 1, pfc2, 1));
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc2.pbias.size()),
                                    &alpha, gfc2bias, 1, pfc2bias, 1));
 #endif
#endif        
    }
};


///////////////////////////////////////////////////////////////////////////////////////////
// Main function

int main(int argc, char **argv)
{

    size_t width, height, channels = 1;

    // Open input data
    printf("Reading input data\n");
    
    // Read dataset sizes
    size_t train_size = ReadUByteDataset(FLAGS_train_images.c_str(), FLAGS_train_labels.c_str(), nullptr, nullptr, width, height);
    size_t test_size = ReadUByteDataset(FLAGS_test_images.c_str(), FLAGS_test_labels.c_str(), nullptr, nullptr, width, height);
    if (train_size == 0)
        return 1;
    
    std::vector<uint8_t> train_images(train_size * width * height * channels), train_labels(train_size);
    std::vector<uint8_t> test_images(test_size * width * height * channels), test_labels(test_size);

    // Read data from datasets
    if (ReadUByteDataset(FLAGS_train_images.c_str(), FLAGS_train_labels.c_str(), &train_images[0], &train_labels[0], width, height) != train_size)
        return 2;
    if (ReadUByteDataset(FLAGS_test_images.c_str(), FLAGS_test_labels.c_str(), &test_images[0], &test_labels[0], width, height) != test_size)
        return 3;

    printf("Done. Training dataset size: %d, Test dataset size: %d\n", (int)train_size, (int)test_size);
    printf("Batch size: %lld, iterations: %d\n", FLAGS_batch_size, FLAGS_iterations);

    // This code snippet saves a random image and its label
    /*
    std::random_device rd_image;
    int random_image = rd_image() % train_size;
    std::stringstream ss; ss << "image-" << (int)train_labels[random_image] << ".pgm";
    SavePGMFile(&train_images[0] + random_image * width*height*channels, width, height, ss.str().c_str());
    */

    // Choose GPU
    int num_gpus;
    checkCudaErrors(cudaGetDeviceCount(&num_gpus));
    if (FLAGS_gpu < 0 || FLAGS_gpu >= num_gpus)
    {
        printf("ERROR: Invalid GPU ID %d (There are %d GPUs on this machine)\n",
               FLAGS_gpu, num_gpus);
        return 4;
    }

    // Create the LeNet network architecture
    ConvBiasLayer conv1((int)channels, 20, 5, (int)width, (int)height);
    MaxPoolLayer pool1(2, 2);
    ConvBiasLayer conv2(conv1.out_channels, 50, 5, conv1.out_width / pool1.stride, conv1.out_height / pool1.stride);
    MaxPoolLayer pool2(2, 2);
    FullyConnectedLayer fc1((conv2.out_channels*conv2.out_width*conv2.out_height) / (pool2.stride * pool2.stride), 
                            500);
    FullyConnectedLayer fc2(fc1.outputs, 10);

    // Initialize CUDNN/CUBLAS training context
    TrainingContext context(FLAGS_gpu, FLAGS_batch_size, conv1, pool1, conv2, pool2, fc1, fc2);
    
#ifdef USE_DROPOUT_LAYER
    float dropRate = 0.4f;
    context.InitDropout(dropRate, FLAGS_batch_size, /*features: */ 1,  /* wid= */  fc1.outputs, /* hei: */  1);
#endif    
    
    // Determine initial network structure
    bool bRet = true;
    if (FLAGS_pretrained)
    {
      bRet = conv1.FromFile("conv1");
      bRet &= conv2.FromFile("conv2");
      bRet &= fc1.FromFile("ip1");
      bRet &= fc2.FromFile("ip2");
    }
    if (!bRet || !FLAGS_pretrained)
    {
        // Create random network
        std::random_device rd;
        std::mt19937 gen(FLAGS_random_seed < 0 ? rd() : static_cast<unsigned int>(FLAGS_random_seed));

        // Xavier weight filling
        float wconv1 = sqrt(3.0f / (conv1.kernel_size * conv1.kernel_size * conv1.in_channels));
        std::uniform_real_distribution<> dconv1(-wconv1, wconv1);
        float wconv2 = sqrt(3.0f / (conv2.kernel_size * conv2.kernel_size * conv2.in_channels));
        std::uniform_real_distribution<> dconv2(-wconv2, wconv2);
        float wfc1 = sqrt(3.0f / (fc1.inputs * fc1.outputs));
        std::uniform_real_distribution<> dfc1(-wfc1, wfc1);
        float wfc2 = sqrt(3.0f / (fc2.inputs * fc2.outputs));
        std::uniform_real_distribution<> dfc2(-wfc2, wfc2);

        // Randomize network
        for (auto&& iter : conv1.pconv)
            iter = static_cast<float>(dconv1(gen));
        for (auto&& iter : conv1.pbias)
            iter = static_cast<float>(dconv1(gen));
        for (auto&& iter : conv2.pconv)
            iter = static_cast<float>(dconv2(gen));
        for (auto&& iter : conv2.pbias)
            iter = static_cast<float>(dconv2(gen));
        for (auto&& iter : fc1.pneurons)
            iter = static_cast<float>(dfc1(gen));
        for (auto&& iter : fc1.pbias)
            iter = static_cast<float>(dfc1(gen));
        for (auto&& iter : fc2.pneurons)
            iter = static_cast<float>(dfc2(gen));
        for (auto&& iter : fc2.pbias)
            iter = static_cast<float>(dfc2(gen));
    }
    
    /////////////////////////////////////////////////////////////////////////////
    // Create GPU data structures    

    // Forward propagation data
    float *d_data, *d_labels, *d_conv1, *d_pool1, *d_conv2, *d_pool2, *d_fc1, *d_fc1relu, *d_fc2, *d_fc2smax;
    //                         Buffer    | Element       | N                   | C                  | H                                 | W
    //-----------------------------------------------------------------------------------------------------------------------------------------
    checkCudaErrors(cudaMalloc(&d_data,    sizeof(float) * context.m_batchSize * channels           * height                            * width));
    checkCudaErrors(cudaMalloc(&d_labels,  sizeof(float) * context.m_batchSize * 1                  * 1                                 * 1));
    checkCudaErrors(cudaMalloc(&d_conv1,   sizeof(float) * context.m_batchSize * conv1.out_channels * conv1.out_height                  * conv1.out_width));
    checkCudaErrors(cudaMalloc(&d_pool1,   sizeof(float) * context.m_batchSize * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride)));
    checkCudaErrors(cudaMalloc(&d_conv2,   sizeof(float) * context.m_batchSize * conv2.out_channels * conv2.out_height                  * conv2.out_width));
    checkCudaErrors(cudaMalloc(&d_pool2,   sizeof(float) * context.m_batchSize * conv2.out_channels * (conv2.out_height / pool2.stride) * (conv2.out_width / pool2.stride)));
    checkCudaErrors(cudaMalloc(&d_fc1,     sizeof(float) * context.m_batchSize * fc1.outputs));    
    checkCudaErrors(cudaMalloc(&d_fc1relu, sizeof(float) * context.m_batchSize * fc1.outputs));
    checkCudaErrors(cudaMalloc(&d_fc2,     sizeof(float) * context.m_batchSize * fc2.outputs));
    checkCudaErrors(cudaMalloc(&d_fc2smax, sizeof(float) * context.m_batchSize * fc2.outputs));    

    float *d_conv1relu, *d_conv2relu;
    checkCudaErrors(cudaMalloc(&d_conv1relu,sizeof(float) * context.m_batchSize * conv1.out_channels * conv1.out_height                          * conv1.out_width));   // same dimension as on  conv1
    checkCudaErrors(cudaMalloc(&d_conv2relu,sizeof(float) * context.m_batchSize * conv2.out_channels * conv2.out_height                        * conv2.out_width));    // same dimension as on  conv2

  
    
    // Network parameters
    float *d_pconv1, *d_pconv1bias, *d_pconv2, *d_pconv2bias;
    float *d_pfc1, *d_pfc1bias, *d_pfc2, *d_pfc2bias;
    
    checkCudaErrors(cudaMalloc(&d_pconv1,     sizeof(float) * conv1.pconv.size()));
    checkCudaErrors(cudaMalloc(&d_pconv1bias, sizeof(float) * conv1.pbias.size()));
    checkCudaErrors(cudaMalloc(&d_pconv2,     sizeof(float) * conv2.pconv.size()));
    checkCudaErrors(cudaMalloc(&d_pconv2bias, sizeof(float) * conv2.pbias.size()));
    checkCudaErrors(cudaMalloc(&d_pfc1,       sizeof(float) * fc1.pneurons.size()));
    checkCudaErrors(cudaMalloc(&d_pfc1bias,   sizeof(float) * fc1.pbias.size()));
    checkCudaErrors(cudaMalloc(&d_pfc2,       sizeof(float) * fc2.pneurons.size()));
    checkCudaErrors(cudaMalloc(&d_pfc2bias,   sizeof(float) * fc2.pbias.size()));    

   
    // Momentum/Adam "v" network parameters
    float *d_vconv1=NULL, *d_vconv1bias=NULL, *d_vconv2=NULL, *d_vconv2bias=NULL;
    float *d_vfc1=NULL, *d_vfc1bias=NULL, *d_vfc2=NULL, *d_vfc2bias=NULL;
    
#if defined(USE_NESTEROV_MOMENTUM) || defined(USE_ADAM)
      checkCudaErrors(cudaMalloc(&d_vconv1, sizeof(float) * conv1.pconv.size()));
      checkCudaErrors(cudaMalloc(&d_vconv1bias, sizeof(float) * conv1.pbias.size()));
      checkCudaErrors(cudaMalloc(&d_vconv2, sizeof(float) * conv2.pconv.size()));
      checkCudaErrors(cudaMalloc(&d_vconv2bias, sizeof(float) * conv2.pbias.size()));
      checkCudaErrors(cudaMalloc(&d_vfc1, sizeof(float) * fc1.pneurons.size()));
      checkCudaErrors(cudaMalloc(&d_vfc1bias, sizeof(float) * fc1.pbias.size()));
      checkCudaErrors(cudaMalloc(&d_vfc2, sizeof(float) * fc2.pneurons.size()));
      checkCudaErrors(cudaMalloc(&d_vfc2bias, sizeof(float) * fc2.pbias.size()));

      FillZeroes<<<RoundUp(conv1.pconv.size(), BW), BW>>>(d_vconv1, conv1.pconv.size());
      FillZeroes<<<RoundUp(conv1.pbias.size(), BW), BW>>>(d_vconv1bias, conv1.pbias.size());
      FillZeroes<<<RoundUp(conv2.pconv.size(), BW), BW>>>(d_vconv2, conv2.pconv.size());
      FillZeroes<<<RoundUp(conv2.pbias.size(), BW), BW>>>(d_vconv2bias, conv2.pbias.size());

      FillZeroes<<<RoundUp(fc1.pneurons.size(), BW), BW>>>(d_vfc1, fc1.pneurons.size());
      FillZeroes<<<RoundUp(fc1.pbias.size(), BW), BW>>>(d_vfc1bias, fc1.pbias.size());
      FillZeroes<<<RoundUp(fc2.pneurons.size(), BW), BW>>>(d_vfc2, fc2.pneurons.size());
      FillZeroes<<<RoundUp(fc2.pbias.size(), BW), BW>>>(d_vfc2bias, fc2.pbias.size());
#endif
        
	
	
    // Adam "m" network parameters
    float *d_mconv1=NULL, *d_mconv1bias=NULL, *d_mconv2=NULL, *d_mconv2bias=NULL;
    float *d_mfc1=NULL, *d_mfc1bias=NULL, *d_mfc2=NULL, *d_mfc2bias=NULL;	

#if defined(USE_ADAM)
      checkCudaErrors(cudaMalloc(&d_mconv1, sizeof(float) * conv1.pconv.size()));
      checkCudaErrors(cudaMalloc(&d_mconv1bias, sizeof(float) * conv1.pbias.size()));
      checkCudaErrors(cudaMalloc(&d_mconv2, sizeof(float) * conv2.pconv.size()));
      checkCudaErrors(cudaMalloc(&d_mconv2bias, sizeof(float) * conv2.pbias.size()));
      checkCudaErrors(cudaMalloc(&d_mfc1, sizeof(float) * fc1.pneurons.size()));
      checkCudaErrors(cudaMalloc(&d_mfc1bias, sizeof(float) * fc1.pbias.size()));
      checkCudaErrors(cudaMalloc(&d_mfc2, sizeof(float) * fc2.pneurons.size()));
      checkCudaErrors(cudaMalloc(&d_mfc2bias, sizeof(float) * fc2.pbias.size()));

      // in the first few time steps the vectors m,v are both initialized and therefore biased at zero, 
      // before they fully “warm up”. 

      FillZeroes<<<RoundUp(conv1.pconv.size(), BW), BW>>>(d_mconv1, conv1.pconv.size());
      FillZeroes<<<RoundUp(conv1.pbias.size(), BW), BW>>>(d_mconv1bias, conv1.pbias.size());
      FillZeroes<<<RoundUp(conv2.pconv.size(), BW), BW>>>(d_mconv2, conv2.pconv.size());
      FillZeroes<<<RoundUp(conv2.pbias.size(), BW), BW>>>(d_mconv2bias, conv2.pbias.size());

      FillZeroes<<<RoundUp(fc1.pneurons.size(), BW), BW>>>(d_mfc1, fc1.pneurons.size());
      FillZeroes<<<RoundUp(fc1.pbias.size(), BW), BW>>>(d_mfc1bias, fc1.pbias.size());
      FillZeroes<<<RoundUp(fc2.pneurons.size(), BW), BW>>>(d_mfc2, fc2.pneurons.size());
      FillZeroes<<<RoundUp(fc2.pbias.size(), BW), BW>>>(d_mfc2bias, fc2.pbias.size());	
#endif
	
	
    
    // Network parameter gradients
    float *d_gconv1, *d_gconv1bias, *d_gconv2, *d_gconv2bias;
    float *d_gfc1, *d_gfc1bias, *d_gfc2, *d_gfc2bias;
    
    checkCudaErrors(cudaMalloc(&d_gconv1,     sizeof(float) * conv1.pconv.size()));
    checkCudaErrors(cudaMalloc(&d_gconv1bias, sizeof(float) * conv1.pbias.size()));
    checkCudaErrors(cudaMalloc(&d_gconv2,     sizeof(float) * conv2.pconv.size()));
    checkCudaErrors(cudaMalloc(&d_gconv2bias, sizeof(float) * conv2.pbias.size()));
    checkCudaErrors(cudaMalloc(&d_gfc1,       sizeof(float) * fc1.pneurons.size()));
    checkCudaErrors(cudaMalloc(&d_gfc1bias,   sizeof(float) * fc1.pbias.size()));    
    checkCudaErrors(cudaMalloc(&d_gfc2,       sizeof(float) * fc2.pneurons.size()));
    checkCudaErrors(cudaMalloc(&d_gfc2bias,   sizeof(float) * fc2.pbias.size()));

    float *d_dconv1relu, *d_dconv2relu; 
    checkCudaErrors(cudaMalloc(&d_dconv1relu,     sizeof(float) * context.m_batchSize * conv1.out_channels * conv1.out_height                         * conv1.out_width)); // same dimension as on d_dpool1
    checkCudaErrors(cudaMalloc(&d_dconv2relu,     sizeof(float) * context.m_batchSize * conv2.out_channels * conv2.out_height                         * conv2.out_width));  // same dimension as on d_dpool2
    
    
    // Differentials w.r.t. data
    float *d_dpool1, *d_dpool2, *d_dconv2, *d_dfc1, *d_dfc1relu, *d_dfc2, *d_dfc2smax, *d_dlossdata;
    //                         Buffer     | Element       | N                   | C                  | H                                 | W
    //-----------------------------------------------------------------------------------------------------------------------------------------
    checkCudaErrors(cudaMalloc(&d_dpool1,   sizeof(float) * context.m_batchSize * conv1.out_channels * conv1.out_height                  * conv1.out_width));
    checkCudaErrors(cudaMalloc(&d_dpool2,   sizeof(float) * context.m_batchSize * conv2.out_channels * conv2.out_height                  * conv2.out_width));
    checkCudaErrors(cudaMalloc(&d_dconv2,   sizeof(float) * context.m_batchSize * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride)));
    checkCudaErrors(cudaMalloc(&d_dfc1,     sizeof(float) * context.m_batchSize * fc1.inputs));
    checkCudaErrors(cudaMalloc(&d_dfc1relu, sizeof(float) * context.m_batchSize * fc1.outputs));
    checkCudaErrors(cudaMalloc(&d_dfc2,     sizeof(float) * context.m_batchSize * fc2.inputs));
    checkCudaErrors(cudaMalloc(&d_dfc2smax, sizeof(float) * context.m_batchSize * fc2.outputs));
    checkCudaErrors(cudaMalloc(&d_dlossdata,sizeof(float) * context.m_batchSize * fc2.outputs));
    
    // Temporary buffers and workspaces
    float *d_onevec;
    void *d_cudnn_workspace = nullptr;    
    checkCudaErrors(cudaMalloc(&d_onevec, sizeof(float)* context.m_batchSize));
    if (context.m_workspaceSize > 0)
        checkCudaErrors(cudaMalloc(&d_cudnn_workspace, context.m_workspaceSize));    

    /////////////////////////////////////////////////////////////////////////////

    // Copy initial network to device
    checkCudaErrors(cudaMemcpyAsync(d_pconv1, &conv1.pconv[0],     sizeof(float) * conv1.pconv.size(),  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pconv1bias, &conv1.pbias[0], sizeof(float) * conv1.pbias.size(),  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pconv2, &conv2.pconv[0],     sizeof(float) * conv2.pconv.size(),  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pconv2bias, &conv2.pbias[0], sizeof(float) * conv2.pbias.size(),  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pfc1, &fc1.pneurons[0],      sizeof(float) * fc1.pneurons.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pfc1bias, &fc1.pbias[0],     sizeof(float) * fc1.pbias.size(),    cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pfc2, &fc2.pneurons[0],      sizeof(float) * fc2.pneurons.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pfc2bias, &fc2.pbias[0],     sizeof(float) * fc2.pbias.size(),    cudaMemcpyHostToDevice));
    
    // Fill one-vector with ones
    FillOnes<<<RoundUp(context.m_batchSize, BW), BW>>>(d_onevec, context.m_batchSize);

    printf("Preparing dataset\n");
    
    // Normalize training set to be in [0,1]
    std::vector<float> train_images_float(train_images.size()), train_labels_float(train_size);
    for (size_t i = 0; i < train_size * channels * width * height; ++i)
        train_images_float[i] = (float)train_images[i] / 255.0f;
    
    for (size_t i = 0; i < train_size; ++i)
        train_labels_float[i] = (float)train_labels[i];

    printf("Training...\n");

    // Use SGD to train the network
    checkCudaErrors(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < FLAGS_iterations; ++iter)
    {
        // Train
        int imageid = iter % (train_size / context.m_batchSize);

        // Prepare current batch on device
        checkCudaErrors(cudaMemcpyAsync(d_data, &train_images_float[imageid * context.m_batchSize * width*height*channels],
                                        sizeof(float) * context.m_batchSize * channels * width * height, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyAsync(d_labels, &train_labels_float[imageid * context.m_batchSize],
                                        sizeof(float) * context.m_batchSize, cudaMemcpyHostToDevice));
        
        
        // Forward propagation
        context.ForwardPropagation(d_data, d_conv1, d_conv1relu, d_pool1, d_conv2, d_conv2relu, d_pool2,  d_fc1, d_fc1relu, d_fc2, d_fc2smax, 
                                   d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
                                   d_cudnn_workspace, d_onevec);

        // Backward propagation
        context.Backpropagation(conv1, pool1, conv2, pool2,
                                d_data, d_labels, d_conv1, d_conv1relu, d_pool1, d_conv2, d_conv2relu, d_pool2,  d_fc1, d_fc1relu, d_fc2, d_fc2smax, d_dlossdata,
                                d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
                                d_gconv1, d_gconv1bias, d_dpool1,  d_dconv1relu,  d_gconv2, d_gconv2bias, d_dconv2, d_dpool2,  d_dconv2relu, d_gfc1, d_gfc1bias, 
                                d_dfc1, d_dfc1relu, d_gfc2, d_gfc2bias, d_dfc2, d_cudnn_workspace, d_onevec);
        
#ifdef USE_SCHEDULED_LEARNING_RATE	    
	float StepDecayScheduleDrop = 0.56;
	float StepDecayScheduleEpochsDrop = 250.0f;
        float learningRate = static_cast<float>(FLAGS_learning_rate * pow(StepDecayScheduleDrop, floor((1.0 + (float)iter) / StepDecayScheduleEpochsDrop)));
#else
        // Compute learning rate  (decaying ~1/T)
        float learningRate = static_cast<float>(FLAGS_learning_rate * pow((1.0 + FLAGS_lr_gamma * iter), (-FLAGS_lr_power)));
#endif
	    
        // Update weights
        context.UpdateWeights(learningRate, conv1, conv2,
                              d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
                              d_gconv1, d_gconv1bias, d_gconv2, d_gconv2bias, d_gfc1, d_gfc1bias, d_gfc2, d_gfc2bias,
			      
                             d_vconv1,      d_vconv1bias,
                             d_vconv2,      d_vconv2bias,
                             d_vfc1,        d_vfc1bias,
                             d_vfc2,        d_vfc2bias,
			      
                             d_mconv1,      d_mconv1bias, 
                             d_mconv2,      d_mconv2bias,
                             d_mfc1,        d_mfc1bias,
                             d_mfc2,        d_mfc2bias);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    printf("Iteration time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / FLAGS_iterations);
    
    if (FLAGS_save_data)
    {
        // Copy trained weights from GPU to CPU
        checkCudaErrors(cudaMemcpy(&conv1.pconv[0], d_pconv1, sizeof(float) * conv1.pconv.size(), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&conv1.pbias[0], d_pconv1bias, sizeof(float) * conv1.pbias.size(), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&conv2.pconv[0], d_pconv2, sizeof(float) * conv2.pconv.size(), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&conv2.pbias[0], d_pconv2bias, sizeof(float) * conv2.pbias.size(), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&fc1.pneurons[0], d_pfc1, sizeof(float) * fc1.pneurons.size(), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&fc1.pbias[0], d_pfc1bias, sizeof(float) * fc1.pbias.size(), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&fc2.pneurons[0], d_pfc2, sizeof(float) * fc2.pneurons.size(), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&fc2.pbias[0], d_pfc2bias, sizeof(float) * fc2.pbias.size(), cudaMemcpyDeviceToHost));
      
        // Now save data
        printf("Saving data to file\n");
        conv1.ToFile("conv1");
        conv2.ToFile("conv2");
        fc1.ToFile("ip1");
        fc2.ToFile("ip2");
    }
    

    float classification_error = 1.0f;

    int classifications = FLAGS_classify;
    if (classifications < 0)
        classifications = (int)test_size;
    
    // Test the resulting neural network's classification
    if (classifications > 0)
    {
        // Initialize a TrainingContext structure for testing (different batch size)
        TrainingContext test_context(FLAGS_gpu, 1, conv1, pool1, conv2, pool2, fc1, fc2);

        // Ensure correct workspaceSize is allocated for testing
        if (context.m_workspaceSize < test_context.m_workspaceSize)
        {
            checkCudaErrors(cudaFree(d_cudnn_workspace));
            checkCudaErrors(cudaMalloc(&d_cudnn_workspace, test_context.m_workspaceSize));
        }

        int num_errors = 0;
        for (int i = 0; i < classifications; ++i)
        {
            std::vector<float> data(width * height);
            // Normalize image to be in [0,1]
            for (int j = 0; j < width * height; ++j)
                data[j] = (float)test_images[i * width*height*channels + j] / 255.0f;

            checkCudaErrors(cudaMemcpyAsync(d_data, &data[0], sizeof(float) * width * height, cudaMemcpyHostToDevice));
            
            // Forward propagate test image
            test_context.ForwardPropagation(d_data, d_conv1, d_conv1relu, d_pool1, d_conv2, d_conv2relu, d_pool2,  d_fc1, d_fc1relu, d_fc2, d_fc2smax, 
                                   d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
                                   d_cudnn_workspace, d_onevec);
            
            // Perform classification
            std::vector<float> class_vec(10);

            // Copy back result
            checkCudaErrors(cudaMemcpy(&class_vec[0], d_fc2smax, sizeof(float) * 10, cudaMemcpyDeviceToHost));

            // Determine classification according to maximal response
            int chosen = 0;
            for (int id = 1; id < 10; ++id)
            {
                if (class_vec[chosen] < class_vec[id]) chosen = id;
            }

            if (chosen != test_labels[i])
                ++num_errors;
        }
        classification_error = (float)num_errors / (float)classifications;

        printf("Classification result: %.2f%% error (used %d images)\n", classification_error * 100.0f, (int)classifications);
    }
        
    // Free data structures
    checkCudaErrors(cudaFree(d_data));
    
    checkCudaErrors(cudaFree(d_conv1));
    checkCudaErrors(cudaFree(d_pool1));
    checkCudaErrors(cudaFree(d_conv1relu));
    checkCudaErrors(cudaFree(d_dconv1relu));
    
    checkCudaErrors(cudaFree(d_conv2));
    checkCudaErrors(cudaFree(d_pool2));
    checkCudaErrors(cudaFree(d_conv2relu));
    checkCudaErrors(cudaFree(d_dconv2relu));
  
    
    checkCudaErrors(cudaFree(d_fc1));
    checkCudaErrors(cudaFree(d_fc1relu));
    checkCudaErrors(cudaFree(d_dfc1relu));
    
    checkCudaErrors(cudaFree(d_fc2));
    checkCudaErrors(cudaFree(d_pconv1));
    checkCudaErrors(cudaFree(d_pconv1bias));
    checkCudaErrors(cudaFree(d_pconv2));
    checkCudaErrors(cudaFree(d_pconv2bias));
    checkCudaErrors(cudaFree(d_pfc1));
    checkCudaErrors(cudaFree(d_pfc1bias));
    checkCudaErrors(cudaFree(d_pfc2));
    checkCudaErrors(cudaFree(d_pfc2bias));
    checkCudaErrors(cudaFree(d_gconv1));
    checkCudaErrors(cudaFree(d_gconv1bias));
    checkCudaErrors(cudaFree(d_gconv2));
    checkCudaErrors(cudaFree(d_gconv2bias));
    checkCudaErrors(cudaFree(d_gfc1));
    checkCudaErrors(cudaFree(d_gfc1bias));
    checkCudaErrors(cudaFree(d_dfc1));
    checkCudaErrors(cudaFree(d_gfc2));
    checkCudaErrors(cudaFree(d_gfc2bias));
    checkCudaErrors(cudaFree(d_dfc2));
    
    
#if defined(USE_NESTEROV_MOMENTUM) || defined(USE_ADAM)
      checkCudaErrors(cudaFree(d_vconv1));
      checkCudaErrors(cudaFree(d_vconv1bias));
      checkCudaErrors(cudaFree(d_vconv2));
      checkCudaErrors(cudaFree(d_vconv2bias));
      checkCudaErrors(cudaFree(d_vfc1));
      checkCudaErrors(cudaFree(d_vfc1bias));
      checkCudaErrors(cudaFree(d_vfc2));
      checkCudaErrors(cudaFree(d_vfc2bias));    
#endif    
	
#if defined(USE_ADAM)
      checkCudaErrors(cudaFree(d_mconv1));
      checkCudaErrors(cudaFree(d_mconv1bias));
      checkCudaErrors(cudaFree(d_mconv2));
      checkCudaErrors(cudaFree(d_mconv2bias));
      checkCudaErrors(cudaFree(d_mfc1));
      checkCudaErrors(cudaFree(d_mfc1bias));
      checkCudaErrors(cudaFree(d_mfc2));
      checkCudaErrors(cudaFree(d_mfc2bias));
#endif
    
    checkCudaErrors(cudaFree(d_dpool1));
    checkCudaErrors(cudaFree(d_dconv2));
    checkCudaErrors(cudaFree(d_dpool2));    
    checkCudaErrors(cudaFree(d_labels));
    checkCudaErrors(cudaFree(d_dlossdata));
    checkCudaErrors(cudaFree(d_onevec));
    if (d_cudnn_workspace != nullptr)
        checkCudaErrors(cudaFree(d_cudnn_workspace));

    return 0;
}

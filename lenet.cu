
// Disclaimer: No Warranty. Use at your own risk.  (See License Information in Root)
// this project was forked from https://github.com/tbennun/cudnn-training

#define CUDNN_MAJOR 7    // CUDA 9.0  cudnn64_7.dll

// =========================================================================================================
// Definitions 

#include "GlobalInits.h"

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

///////////////////////////////////////////////////////////////////////////////////////////
// Main function

int main(int argc, char **argv)
{
  int flags = 8;  // select Nadam  (YET FOR TEST)
  #define LEARNING_RATE2  0.001
  UpdateGlobalParameters(BATCH_SIZE, BW, ITERATIONS, LEARNING_RATE2,
                                           LEARNING_RATE_POLICY_GAMMA, LEARNING_RATE_POLICY_POWER,
                                           StepDecayScheduleDrop, StepDecayScheduleEpochsDrop,
                                           ExponentialDecayK,
                                           MOMENTUM, DROP_RATE, flags);	

	

   size_t width, height, channels = 1;

   ShowGlobalParameterSettings();

	
        
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
    
    float dropRate = (float)FLAGS_drop_rate;  // NOTE: if drop rate is 0.0 internally drop out layer is not applied
    context.InitDropout(dropRate, FLAGS_batch_size, /*features: */ 1,  /* wid= */  fc1.outputs, /* hei: */  1);
	
    
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
        for (auto&& iter : conv1.pneurons)
            iter = static_cast<float>(dconv1(gen));
        for (auto&& iter : conv1.pbias)
            iter = static_cast<float>(dconv1(gen));
        for (auto&& iter : conv2.pneurons)
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
    
    checkCudaErrors(cudaMalloc(&d_pconv1,     sizeof(float) * conv1.pneurons.size()));
    checkCudaErrors(cudaMalloc(&d_pconv1bias, sizeof(float) * conv1.pbias.size()));
    checkCudaErrors(cudaMalloc(&d_pconv2,     sizeof(float) * conv2.pneurons.size()));
    checkCudaErrors(cudaMalloc(&d_pconv2bias, sizeof(float) * conv2.pbias.size()));
    checkCudaErrors(cudaMalloc(&d_pfc1,       sizeof(float) * fc1.pneurons.size()));
    checkCudaErrors(cudaMalloc(&d_pfc1bias,   sizeof(float) * fc1.pbias.size()));
    checkCudaErrors(cudaMalloc(&d_pfc2,       sizeof(float) * fc2.pneurons.size()));
    checkCudaErrors(cudaMalloc(&d_pfc2bias,   sizeof(float) * fc2.pbias.size()));    


   
    // Momentum/Adam "v" network parameters
    float *d_vconv1=NULL, *d_vconv1bias=NULL, *d_vconv2=NULL, *d_vconv2bias=NULL;
    float *d_vfc1=NULL, *d_vfc1bias=NULL, *d_vfc2=NULL, *d_vfc2bias=NULL;
    
    if ((FLAGS_momentum > 0.0) || Adam)
    {
      checkCudaErrors(cudaMalloc(&d_vconv1, sizeof(float) * conv1.pneurons.size()));
      checkCudaErrors(cudaMalloc(&d_vconv1bias, sizeof(float) * conv1.pbias.size()));
      checkCudaErrors(cudaMalloc(&d_vconv2, sizeof(float) * conv2.pneurons.size()));
      checkCudaErrors(cudaMalloc(&d_vconv2bias, sizeof(float) * conv2.pbias.size()));
      checkCudaErrors(cudaMalloc(&d_vfc1, sizeof(float) * fc1.pneurons.size()));
      checkCudaErrors(cudaMalloc(&d_vfc1bias, sizeof(float) * fc1.pbias.size()));
      checkCudaErrors(cudaMalloc(&d_vfc2, sizeof(float) * fc2.pneurons.size()));
      checkCudaErrors(cudaMalloc(&d_vfc2bias, sizeof(float) * fc2.pbias.size()));

      FillZeroes<<<RoundUp(conv1.pneurons.size(), BW), BW>>>(d_vconv1, conv1.pneurons.size());
      FillZeroes<<<RoundUp(conv1.pbias.size(), BW), BW>>>(d_vconv1bias, conv1.pbias.size());
      FillZeroes<<<RoundUp(conv2.pneurons.size(), BW), BW>>>(d_vconv2, conv2.pneurons.size());
      FillZeroes<<<RoundUp(conv2.pbias.size(), BW), BW>>>(d_vconv2bias, conv2.pbias.size());

      FillZeroes<<<RoundUp(fc1.pneurons.size(), BW), BW>>>(d_vfc1, fc1.pneurons.size());
      FillZeroes<<<RoundUp(fc1.pbias.size(), BW), BW>>>(d_vfc1bias, fc1.pbias.size());
      FillZeroes<<<RoundUp(fc2.pneurons.size(), BW), BW>>>(d_vfc2, fc2.pneurons.size());
      FillZeroes<<<RoundUp(fc2.pbias.size(), BW), BW>>>(d_vfc2bias, fc2.pbias.size());
    }
        
	
	
    // Adam "m" network parameters
    float *d_mconv1=NULL, *d_mconv1bias=NULL, *d_mconv2=NULL, *d_mconv2bias=NULL;
    float *d_mfc1=NULL, *d_mfc1bias=NULL, *d_mfc2=NULL, *d_mfc2bias=NULL;	

    if (Adam || Nadam)
    {
      checkCudaErrors(cudaMalloc(&d_mconv1,     sizeof(float) * conv1.pneurons.size()));
      checkCudaErrors(cudaMalloc(&d_mconv1bias, sizeof(float) * conv1.pbias.size()));
      checkCudaErrors(cudaMalloc(&d_mconv2,     sizeof(float) * conv2.pneurons.size()));
      checkCudaErrors(cudaMalloc(&d_mconv2bias, sizeof(float) * conv2.pbias.size()));
      checkCudaErrors(cudaMalloc(&d_mfc1,       sizeof(float) * fc1.pneurons.size()));
      checkCudaErrors(cudaMalloc(&d_mfc1bias,   sizeof(float) * fc1.pbias.size()));
      checkCudaErrors(cudaMalloc(&d_mfc2,       sizeof(float) * fc2.pneurons.size()));
      checkCudaErrors(cudaMalloc(&d_mfc2bias,   sizeof(float) * fc2.pbias.size()));

      // in the first few time steps the vectors m,v are both initialized and therefore biased at zero, 
      // before they fully “warm up”. 

      FillZeroes<<<RoundUp(conv1.pneurons.size(), BW), BW>>>(d_mconv1, conv1.pneurons.size());
      FillZeroes<<<RoundUp(conv1.pbias.size(), BW), BW>>>(d_mconv1bias, conv1.pbias.size());
      FillZeroes<<<RoundUp(conv2.pneurons.size(), BW), BW>>>(d_mconv2, conv2.pneurons.size());
      FillZeroes<<<RoundUp(conv2.pbias.size(), BW), BW>>>(d_mconv2bias, conv2.pbias.size());

      FillZeroes<<<RoundUp(fc1.pneurons.size(), BW), BW>>>(d_mfc1, fc1.pneurons.size());
      FillZeroes<<<RoundUp(fc1.pbias.size(), BW), BW>>>(d_mfc1bias, fc1.pbias.size());
      FillZeroes<<<RoundUp(fc2.pneurons.size(), BW), BW>>>(d_mfc2, fc2.pneurons.size());
      FillZeroes<<<RoundUp(fc2.pbias.size(), BW), BW>>>(d_mfc2bias, fc2.pbias.size());
    }
	
	
    
    // Network parameter gradients
    float *d_gconv1, *d_gconv1bias, *d_gconv2, *d_gconv2bias;
    float *d_gfc1, *d_gfc1bias, *d_gfc2, *d_gfc2bias;
    
    checkCudaErrors(cudaMalloc(&d_gconv1,       sizeof(float) * conv1.pneurons.size()));
    checkCudaErrors(cudaMalloc(&d_gconv1bias,   sizeof(float) * conv1.pbias.size()));
    checkCudaErrors(cudaMalloc(&d_gconv2,       sizeof(float) * conv2.pneurons.size()));
    checkCudaErrors(cudaMalloc(&d_gconv2bias,   sizeof(float) * conv2.pbias.size()));
    checkCudaErrors(cudaMalloc(&d_gfc1,         sizeof(float) * fc1.pneurons.size()));
    checkCudaErrors(cudaMalloc(&d_gfc1bias,     sizeof(float) * fc1.pbias.size()));    
    checkCudaErrors(cudaMalloc(&d_gfc2,         sizeof(float) * fc2.pneurons.size()));
    checkCudaErrors(cudaMalloc(&d_gfc2bias,     sizeof(float) * fc2.pbias.size()));
	
	
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
    checkCudaErrors(cudaMemcpyAsync(d_pconv1, &conv1.pneurons[0],       sizeof(float) * conv1.pneurons.size(),  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pconv1bias, &conv1.pbias[0],      sizeof(float) * conv1.pbias.size(),   cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pconv2, &conv2.pneurons[0],       sizeof(float) * conv2.pneurons.size(),  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pconv2bias, &conv2.pbias[0],      sizeof(float) * conv2.pbias.size(),   cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pfc1, &fc1.pneurons[0],           sizeof(float) * fc1.pneurons.size(),  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pfc1bias, &fc1.pbias[0],          sizeof(float) * fc1.pbias.size(),        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pfc2, &fc2.pneurons[0],           sizeof(float) * fc2.pneurons.size(),  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pfc2bias, &fc2.pbias[0],          sizeof(float) * fc2.pbias.size(),        cudaMemcpyHostToDevice));


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
    float momentum = (float)FLAGS_momentum;	
    for (int iteration = 0; iteration < FLAGS_iterations; ++iteration)
    {
        // Train
        int imageid = iteration % (train_size / context.m_batchSize);  // this wraps around to the beginning of the data set in case of the last batch block

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
                                d_data, d_labels, d_conv1, d_conv1relu, d_pool1,  d_conv2, d_conv2relu, d_pool2,  d_fc1, d_fc1relu, d_fc2, d_fc2smax, d_dlossdata,
                                d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
                                d_gconv1, d_gconv1bias, d_dpool1, d_dconv1relu,  d_gconv2, d_gconv2bias, d_dconv2, d_dpool2,  d_dconv2relu, d_gfc1, d_gfc1bias, 
                                d_dfc1, d_dfc1relu, d_gfc2, d_gfc2bias, d_dfc2, d_cudnn_workspace, d_onevec);

	    
        // Compute learning rate
        float learningRate;
#if 0
        if (PolicySteps)
        {
          int iStep = 0;
          while ( iStep < PolicySteps)
          {
            if (iteration >= LearningRatePolicy[iStep].iterationStep) break;
            ++iStep;
          }
          if (iStep >= PolicySteps)  iStep = PolicySteps - 1;
          learningRate = LearningRatePolicy[iStep].LearningRate;
        }
        else 
#endif
       if (ConstantLearningRate)
       {
         learningRate = static_cast<float>(FLAGS_learning_rate);
       }
       else if  (ExponentialDecayK != 0.0)   // 0.0 = OFF
       {
            // Exponential Decay
           learningRate = static_cast<float>(FLAGS_learning_rate * exp(-ExponentialDecayK * (float)iteration));
       }
       else if (StepDecayScheduleDrop)    // 0.0 = OFF
       {
            // Step decay schedule drops the learning rate by a factor every few epochs.
            learningRate = static_cast<float>(FLAGS_learning_rate * pow(StepDecayScheduleDrop, floor((1.0 + (float)iteration) / StepDecayScheduleEpochsDrop)));
       }
       else // default "inv" method   (the learning rate is decaying as ~1/T)
       {
          learningRate = static_cast<float>(FLAGS_learning_rate * pow((1.0 + FLAGS_lr_gamma * iteration), (-FLAGS_lr_power)));
       }
	    

	    
        // Update weights
        context.UpdateWeights(learningRate, momentum, conv1, conv2,
                              d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
                              d_gconv1, d_gconv1bias, d_gconv2, d_gconv2bias, d_gfc1, d_gfc1bias, d_gfc2, d_gfc2bias,
			      
                             d_vconv1,  d_vconv1bias,    // for Nesterov Momentum + Adam + Adamax + Nadam + Nadamax
                             d_vconv2,  d_vconv2bias,
                             d_vfc1,    d_vfc1bias,
                             d_vfc2,    d_vfc2bias,

                             d_mconv1,  d_mconv1bias,  // for Adam + Nadam + Adamax + Nadam + Nadamax
                             d_mconv2,  d_mconv2bias,
                             d_mfc1,    d_mfc1bias,
                             d_mfc2,    d_mfc2bias);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    printf("Iteration time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / FLAGS_iterations);
    
    if (FLAGS_save_data)
    {
        // Copy trained weights from GPU to CPU
        checkCudaErrors(cudaMemcpy(&conv1.pneurons[0], d_pconv1, sizeof(float) * conv1.pneurons.size(), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&conv1.pbias[0], d_pconv1bias, sizeof(float) * conv1.pbias.size(), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&conv2.pneurons[0], d_pconv2, sizeof(float) * conv2.pneurons.size(), cudaMemcpyDeviceToHost));
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
    checkCudaErrors(cudaFree(d_conv1relu));
    checkCudaErrors(cudaFree(d_dconv1relu));
    checkCudaErrors(cudaFree(d_pool1));    
    checkCudaErrors(cudaFree(d_conv2));
    checkCudaErrors(cudaFree(d_conv2relu));
    checkCudaErrors(cudaFree(d_dconv2relu));
    checkCudaErrors(cudaFree(d_pool2));
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

    if ((FLAGS_momentum > 0.0) || Adam)
    {
      checkCudaErrors(cudaFree(d_vconv1));
      checkCudaErrors(cudaFree(d_vconv1bias));
      checkCudaErrors(cudaFree(d_vconv2));
      checkCudaErrors(cudaFree(d_vconv2bias));
      checkCudaErrors(cudaFree(d_vfc1));
      checkCudaErrors(cudaFree(d_vfc1bias));
      checkCudaErrors(cudaFree(d_vfc2));
      checkCudaErrors(cudaFree(d_vfc2bias));
    }

    if (Adam || Nadam)
    {
      checkCudaErrors(cudaFree(d_mconv1));
      checkCudaErrors(cudaFree(d_mconv1bias));
      checkCudaErrors(cudaFree(d_mconv2));
      checkCudaErrors(cudaFree(d_mconv2bias));
      checkCudaErrors(cudaFree(d_mfc1));
      checkCudaErrors(cudaFree(d_mfc1bias));
      checkCudaErrors(cudaFree(d_mfc2));
      checkCudaErrors(cudaFree(d_mfc2bias));
    }


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

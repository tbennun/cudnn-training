#pragma once

// Disclaimer: No Warranty. Use at your own risk.  (See License Information in Root)


//////////////////////////////////////////////////////////////////////////////
// includes


#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cfloat>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cudnn.h>





//////////////////////////////////////////////////////////////////////////////
// Error handling
// Adapted from the CUDNN classification code 
// sample: https://developer.nvidia.com/cuDNN

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)



//////////////////////////////////////////////////////////////////////////////

/**
 * Computes ceil(x / y) for integral nonnegative values.
 */
static inline unsigned int RoundUp(unsigned int nominator, unsigned int denominator)
{
    return (nominator + denominator - 1) / denominator;
}


//////////////////////////////////////////////////////////////////////////////


#ifdef ALLOW_SAVE_AS_PGM_FILE
// from https://github.com/tbennun/cudnn-training
/**
 * Saves a PGM grayscale image out of unsigned 8-bit data
 */
void SavePGMFile(const unsigned char *data, size_t width, size_t height, const char *filename)
{
    FILE *fp = fopen(filename, "wb");
    if (fp)
    {
        //fprintf(fp, "P5\n%lu %lu\n255\n", width, height);
        fprintf(fp, "P5\n%llu %llu\n255\n", width, height);
        fwrite(data, sizeof(unsigned char), width * height, fp);
        fclose(fp);
    }
}
#endif




int  UpdateGlobalParameters(int flags)
{

  // flags bit 0: set=Use Adam
  // flags bit 1: set=Use Adamax
  // flags bit 2: set=Use Nadam
  // flags bit 3: set=Use Nadamax 
  // flags bit 8: set=constant learning rate (clear= default: decaying as ~1/T)
  if (flags & 1)  Adam = true; else Adam = false;
  if (flags & 2)  Adamax = true; else Adamax = false;
  if (flags & 4)  Nadam = true; else Nadam = false;
  if (flags & 8)  Nadamax = true; else Nadamax = false;
  if (flags & 256)  ConstantLearningRate = true; else ConstantLearningRate = false;
  


  if (Adamax) Adam = true;  // all basics are required on both Adam and Adamax
  if (Nadam || Nadamax)
  {
    Adam = true;  // all basics are required on both Adam and Adamax
    if (FLAGS_momentum == 0.0)  { Nadam = false;  Nadamax = false;} // fall-through to basic Adam if no momentum value present
  }
  
  

  if (batchSize)
  {
    FLAGS_batch_size = batchSize;
    BW = batchSize * 2;
  }
  else
  {
    FLAGS_batch_size = BATCH_SIZE;
    BW = defaultBW;
  };

  if (userBW)
    BW = userBW;

  if (iterations)
    FLAGS_iterations = iterations;
  else
    FLAGS_iterations = ITERATIONS;

  if (LearningRate != 0.0f)
    FLAGS_learning_rate = LearningRate;
  else
    FLAGS_learning_rate = LEARNING_RATE;


  if (LearningRatePolicyGamma)
     FLAGS_lr_gamma = LearningRatePolicyGamma;
  else
     FLAGS_lr_gamma = LEARNING_RATE_POLICY_GAMMA;

  if (LearningRatePolicyPower)
    FLAGS_lr_power = LearningRatePolicyPower;
  else
    FLAGS_lr_power = LEARNING_RATE_POLICY_POWER;


    FLAGS_momentum = Momentum;   // 0.0 here means: simply use pure SDG
    FLAGS_drop_rate = DropRate;  // 0.0 here means: simply avoid DropOut Layer

    StepDecayScheduleDrop = DecayScheduleDrop; //  0.5; //  0.0=OFF
    StepDecayScheduleEpochsDrop = DecayScheduleEpochsDrop;  // 250.0f;

    ExponentialDecayK = expDecayK;

}


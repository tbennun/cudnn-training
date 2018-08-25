#pragma once

// Disclaimer: No Warranty. Use at your own risk.  (See License Information in Root)


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


// Constant versions of gflags   (see https://github.com/gflags/gflags)  (formerly Google Commandline Flags)
#define DEFINE_int32(flag, default_value, description) const int FLAGS_##flag = (default_value)
#define DEFINE_uint64(flag, default_value, description) const unsigned long long FLAGS_##flag = (default_value)
#define DEFINE_bool(flag, default_value, description) const bool FLAGS_##flag = (default_value)
#define DEFINE_double(flag, default_value, description) const double FLAGS_##flag = (default_value)
#define DEFINE_string(flag, default_value, description) const std::string FLAGS_##flag ((default_value))




///////////////////////////////////////////////////////////////////////////////////////////
// Command-line flags

// Application parameters
DEFINE_int32(gpu, 0, "The GPU ID to use");
//DEFINE_int32(iterations, 1000, "Number of iterations for training");
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

#if 0
#define MAX_POLICY_STEPS  1000
struct PolicyStep
{
  int iterationStep;
  double LearningRate;
};
int PolicySteps = 4;   // 0=OFF
/*
PolicyStep LearningRatePolicy[MAX_POLICY_STEPS] = 
{ 
    { 0, 0.001f  },
    { 100, 0.0005f  },
    { 500, 0.0003f  },
    { 750, 0.0001f  }
};
*/
/*
PolicyStep LearningRatePolicy[MAX_POLICY_STEPS] = 
{ 
    { 0, 0.001f  },
    { 100, 0.0004f  },
    { 500, 0.0002f  },
    { 750, 0.00005f  }
};
*/
/*
PolicyStep LearningRatePolicy[MAX_POLICY_STEPS] = 
{ 
    { 0, 0.001f  },
    { 100, 0.0005f  },
    { 400, 0.0003f  },
    { 750, 0.00007f  }
};
*/
PolicyStep LearningRatePolicy[MAX_POLICY_STEPS] = 
{ 
    { 0, 0.001f  },
    { 100, 0.0004f  },
    { 500, 0.0002f  },
    { 750, 0.00005f  }
};
#endif

double ExponentialDecayK = 0.001f; 

double StepDecayScheduleDrop = 0.56; //  0.0=OFF
double StepDecayScheduleEpochsDrop = 250.0f;

// Batch parameters
//DEFINE_uint64(batch_size, 64, "Batch size for training");
//DEFINE_uint64(batch_size, BATCH_SIZE , "Batch size for training");

    // With compute capability 3.0 or higher, you can have up to 2^32 - 1 blocks in the x-dimension, 
    // and at most 65535 blocks in the y and z dimensions. See Table H.1. 
    // Feature Support per Compute Capability of the CUDA C Programming Guide Version 9.1.

DEFINE_bool(pretrained, false, "Use the pretrained CUDNN model as input");
DEFINE_bool(save_data, false, "Save pretrained weights to file");







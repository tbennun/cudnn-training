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



#pragma once

// Disclaimer: No Warranty. Use at your own risk.  (See License Information in Root)

///////////////////////////////////////////////////////////////////////////////////////////
// GPU Kernels

// from https://github.com/tbennun/cudnn-training
/**
 * Fills a floating-point array with ones.
 *
 * @param vec The array to fill.
 * @param size The number of elements in the array.
 */
__global__ void FillOnes(float *vec, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    vec[idx] = 1.0f;
}



__global__ void FillZeroes(float *vec, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    vec[idx] = 0.0f;
}


// from https://github.com/tbennun/cudnn-training
/**
 * Computes the backpropagation results of the Softmax loss for each result in a batch.
 * Uses the softmax values obtained from forward propagation to compute the difference.
 *
 * @param label The training batch label values.
 * @param num_labels The number of possible labels.
 * @param batch_size The size of the trained batch.
 * @param diff The resulting gradient.
 */
__global__ void SoftmaxLossBackprop(const float *label, int num_labels, int batch_size, float *diff)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size)
        return;

    const int label_value = static_cast<int>(label[idx]);

    // For each item in the batch, decrease the result of the label's value by 1
    diff[idx * num_labels + label_value] -= 1.0f;
}




/**
 * Calculate SGD Momentum Weight Update 
 *
 // SGD momentum
      // mu= Momentum update    v=momentum=0.9
      // v[i] = mu * v[i] - learning_rate * dx # integrate velocity
      // x += v # integrate position
        const float MomentumUpdate = 0.9f;


     // v[i] = mu*v[i] + learning_rate * grad[i]

 */
__global__ void SGDmomentumWeightUpdate(float *weights,  float *gradients, float *v, float learning_rate, float MomentumUpdate, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    v[idx] = MomentumUpdate * v[idx] + learning_rate * gradients[idx];
    weights[idx] += v[idx];

#if 0
    // pure SGD:
    float  v0  = learning_rate * gradients[idx];
    weights[idx] += v0;
#endif
}





__global__ void NesterovMomentumWeightUpdate(float *weights,  float *gradients, float *v, float learning_rate,  float MomentumUpdate,   int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    
       //temp = learning_rate * gradient[i]  
       // // step back then over-step  (as in Caffe)
       // v[i] = (1 + momentum) * ( v[i]+temp)  ) - temp;
       // weights[i] += v[i]

/*
   // learning_rate = - learning_rate;
  //  learning_rate = 0.001f;
    const float MomentumUpdate = 0.9f;
    const float temp =  (learning_rate * gradients[idx]);
    v[idx] = ((1.0f + MomentumUpdate) * (v[idx] - temp)) + temp;
    weights[idx] += v[idx];
*/

//v_prev = v # back this up
//v = mu * v - learning_rate * dx # velocity update stays the same
//x += -mu * v_prev + (1 + mu) * v # position update changes form

#if 1
    float v_prev = v[idx];
   // const float MomentumUpdate = 0.9f;
    v[idx] = MomentumUpdate * v[idx] + learning_rate * gradients[idx];      // here + learning_rate (cause its already negated)
    weights[idx] += -MomentumUpdate * v_prev + (1.0f + MomentumUpdate) * v[idx];
#endif


#if 0
  //  const float MomentumUpdate = 0.9f;
    v[idx] = MomentumUpdate * v[idx] + learning_rate * gradients[idx];
    weights[idx] += v[idx];
#endif

#if 0
    // pure SGD:
    float  v0  = learning_rate * gradients[idx];
    weights[idx] += v0;
#endif
}





__global__ void AdamWeightUpdate(float *weights, float *gradients, float *v, float *m, float learning_rate, float beta1, float beta2, float beta1_t, float beta2_t,  int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;


  // http://cs231n.github.io/neural-networks-3/
  //m = beta1*m + (1-beta1)*dx
  //mt = m / (1-beta1**t)
  //v = beta2*v + (1-beta2)*(dx**2)
  //vt = v / (1-beta2**t)
  //x += - learning_rate * mt / (np.sqrt(vt) + eps)

  float dx = gradients[idx];
  m[idx] = beta1 * m[idx] + (1 - beta1) * dx;
  float mt = m[idx] / (1 - beta1_t);
  v[idx] = beta2 * v[idx] + (1 - beta2) * (dx * dx);
  float vt = v[idx] / (1 - beta2_t);

  const float eps = 1e-8;
  weights[idx] += learning_rate * mt / (sqrt(vt) + eps);
    // NOTE: learning_rate is already negated


}





__global__ void AdamaxWeightUpdate(float *weights, float *gradients, float *v, float *m, float learning_rate, float beta1, float beta2, float beta1_t,  int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;


  float dx = gradients[idx];
  m[idx] = beta1 * m[idx] + (1 - beta1) * dx;
  float mt = m[idx] / (1 - beta1_t);
  v[idx] = max(beta2 * v[idx], + abs(dx));
  float vt = v[idx]; //  / (1 - beta2_t);

  const float eps = 1e-8;
  weights[idx] += learning_rate * mt / (vt + eps);
    // NOTE: learning_rate is already negated

}





__global__ void NadamWeightUpdate(float *weights, float *gradients, float *v, float *m,   float learning_rate,  float MomentumUpdate, float NextMomentumUpdate,  float beta2, float beta1_t,  float beta2_t,  int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  
   // NOTE: beta1 is the MomentumUpdate
  //             beta1_t is accumulated multiplied MomentumUpdate

  float dx = gradients[idx];

   m[idx] = MomentumUpdate * m[idx] + (1 - MomentumUpdate) *  dx;   //     m[idx] = beta1 * m[idx] + (1 - beta1) * dx;

  float mt = ( NextMomentumUpdate * m[idx] / (1 - (beta1_t * NextMomentumUpdate)) )
                  + ( (1 - MomentumUpdate) * dx / (1 - beta1_t) );                     //       float mt = m[idx] / (1 - beta1_t);

  v[idx] = beta2 * v[idx] + (1 - beta2) * (dx * dx);
  float vt = v[idx] / (1 - beta2_t);

  const float eps = 1e-8;
  weights[idx] += learning_rate * mt / (sqrt(vt) + eps);
    // NOTE: learning_rate is already negated

}





__global__ void NadamaxWeightUpdate(float *weights, float *gradients, float *v, float *m,   float learning_rate,  float MomentumUpdate, float NextMomentumUpdate,  float beta2, float beta1_t,  float beta2_t,  int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  
   // NOTE: beta1 is the MomentumUpdate
  //             beta1_t is accumulated multiplied MomentumUpdate

  float dx = gradients[idx];

   m[idx] = MomentumUpdate * m[idx] + (1 - MomentumUpdate) *  dx;   //     m[idx] = beta1 * m[idx] + (1 - beta1) * dx;

  float mt = ( NextMomentumUpdate * m[idx] / (1 - (beta1_t * NextMomentumUpdate)) )
                  + ( (1 - MomentumUpdate) * dx / (1 - beta1_t) );                     //       float mt = m[idx] / (1 - beta1_t);


  v[idx] = max(beta2 * v[idx], + abs(dx));
  float vt = v[idx] / (1 - beta2_t);

  const float eps = 1e-8;
  weights[idx] += learning_rate * mt / (sqrt(vt) + eps);
    // NOTE: learning_rate is already negated

}

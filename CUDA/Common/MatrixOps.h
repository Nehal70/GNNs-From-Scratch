#pragma once

#include <cuda_runtime.h>

namespace CUDA_ML {

class MatrixOps {
public:
    // Matrix multiplication with bias addition: C = A * B + bias
    static void matmulAddBias(const float* A, const float* B, const float* bias,
                             float* C, int batch_size, int input_size, int output_size);
    
    // Matrix multiplication with transpose: C = A * B^T
    static void matmulTranspose(const float* A, const float* B, float* C,
                              int batch_size, int input_size, int output_size);
    
    // Element-wise operations
    static void subtract(const float* A, const float* B, float* C, int size);
    static void addBias(const float* input, const float* bias, int size);
    
    // Loss and reduction operations
    static void squaredError(const float* y_true, const float* y_pred, float* error, int size);
    static void sumReduce(const float* input, float* output, int size);
    static void sumRows(const float* input, float* output, int rows, int cols);
    
    // Parameter updates
    static void updateWeights(float* weights, const float* gradients, 
                             float learning_rate, int size);
    static void updateBiases(float* biases, const float* gradients,
                            float learning_rate, int size);
};

} // namespace CUDA_ML




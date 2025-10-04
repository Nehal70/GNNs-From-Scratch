#include "MatrixOps.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace CUDA_ML {

// CUDA kernel for matrix multiplication with bias addition
__global__ void matmulAddBiasKernel(const float* A, const float* B, const float* bias,
                                   float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum + bias[col];
    }
}

// CUDA kernel for matrix transpose multiplication
__global__ void matmulTransposeKernel(const float* A, const float* B, float* C,
                                     int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[col * k + i];  // B is transposed
        }
        C[row * n + col] = sum;
    }
}

// CUDA kernel for element-wise subtraction
__global__ void subtractKernel(const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        C[idx] = A[idx] - B[idx];
    }
}

// CUDA kernel for adding bias
__global__ void addBiasKernel(const float* input, const float* bias, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = input[idx] + bias[idx % (size / (blockIdx.x + 1))];
    }
}

// CUDA kernel for squared error calculation
__global__ void squaredErrorKernel(const float* y_true, const float* y_pred, float* error, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float diff = y_true[idx] - y_pred[idx];
        error[idx] = diff * diff;
    }
}

// CUDA kernel for sum reduction
__global__ void sumReduceKernel(const float* input, float* output, int size) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < size) ? input[i] : 0.0f;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// CUDA kernel for sum over rows
__global__ void sumRowsKernel(const float* input, float* output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < cols) {
        float sum = 0.0f;
        for (int row = 0; row < rows; ++row) {
            sum += input[row * cols + col];
        }
        output[col] = sum;
    }
}

// CUDA kernel for updating weights
__global__ void updateWeightsKernel(float* weights, const float* gradients, 
                                   float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}

// CUDA kernel for updating biases
__global__ void updateBiasesKernel(float* biases, const float* gradients,
                                  float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        biases[idx] -= learning_rate * gradients[idx];
    }
}

void MatrixOps::matmulAddBias(const float* A, const float* B, const float* bias,
                             float* C, int batch_size, int input_size, int output_size) {
    dim3 blockSize(16, 16);
    dim3 gridSize((output_size + blockSize.x - 1) / blockSize.x,
                  (batch_size + blockSize.y - 1) / blockSize.y);
    
    matmulAddBiasKernel<<<gridSize, blockSize>>>(
        A, B, bias, C, batch_size, output_size, input_size
    );
    
    cudaDeviceSynchronize();
}

void MatrixOps::matmulTranspose(const float* A, const float* B, float* C,
                              int batch_size, int input_size, int output_size) {
    dim3 blockSize(16, 16);
    dim3 gridSize((output_size + blockSize.x - 1) / blockSize.x,
                  (batch_size + blockSize.y - 1) / blockSize.y);
    
    matmulTransposeKernel<<<gridSize, blockSize>>>(
        A, B, C, batch_size, output_size, input_size
    );
    
    cudaDeviceSynchronize();
}

void MatrixOps::subtract(const float* A, const float* B, float* C, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    subtractKernel<<<numBlocks, blockSize>>>(A, B, C, size);
    cudaDeviceSynchronize();
}

void MatrixOps::addBias(const float* input, const float* bias, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    addBiasKernel<<<numBlocks, blockSize>>>(input, bias, input, size);
    cudaDeviceSynchronize();
}

void MatrixOps::squaredError(const float* y_true, const float* y_pred, float* error, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    squaredErrorKernel<<<numBlocks, blockSize>>>(y_true, y_pred, error, size);
    cudaDeviceSynchronize();
}

void MatrixOps::sumReduce(const float* input, float* output, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    sumReduceKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(input, output, size);
    cudaDeviceSynchronize();
}

void MatrixOps::sumRows(const float* input, float* output, int rows, int cols) {
    int blockSize = 256;
    int numBlocks = (cols + blockSize - 1) / blockSize;
    
    sumRowsKernel<<<numBlocks, blockSize>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}

void MatrixOps::updateWeights(float* weights, const float* gradients, 
                             float learning_rate, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    updateWeightsKernel<<<numBlocks, blockSize>>>(weights, gradients, learning_rate, size);
    cudaDeviceSynchronize();
}

void MatrixOps::updateBiases(float* biases, const float* gradients,
                            float learning_rate, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    updateBiasesKernel<<<numBlocks, blockSize>>>(biases, gradients, learning_rate, size);
    cudaDeviceSynchronize();
}

} // namespace CUDA_ML




// LinearRegression.cu
#include "LinearRegression.h"
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// kernel for calculating the gradients for slope and intercept
__global__ void calculateGradientsKernel(const float* x, const float* y, float* slope_gradient, float* intercept_gradient, float current_slope, float current_intercept, int n) {

    // computes the global index for the thread and assigns it to idx
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // ensures that the thread is less than the maximum size of the dataset
    if (idx < n) {
        // parallely computes the prediction and error for each sample in the dataset
        float prediction = current_slope * x[idx] + current_intercept;
        float error = prediction - y[idx];
        
        // parallely computes the gradient for slope and intercept based on error computed for each sample and input x value
        slope_gradient[idx] = error * x[idx];
        intercept_gradient[idx] = error;
    }
}

// kernel used during predictions on test data
__global__ void predictKernel(const float* x, float* predictions, float slope, float intercept, int n) {

    // computes the global index for the thread and assigns it to idx
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // ensures that the thread is less than the maximum size of the dataset
    if (idx < n) {

        // computes the prediction for each test sample based on the final slope and intercept post training
        predictions[idx] = slope * x[idx] + intercept;
    }
}

// constructor function
LinearRegression::LinearRegression(float learning_rate) 
    : learning_rate_(learning_rate), slope_(0.0f), intercept_(0.0f), is_trained_(false) {
    
    std::cout << "linear regression initialized with learning rate: " << learning_rate_ << std::endl;
}

// training function
void LinearRegression::train(const std::vector<float>& x_train, const std::vector<float>& y_train) {
    if (x_train.size() != y_train.size()) {
        throw std::invalid_argument("training data sizes must match");
    }
    
    int n = x_train.size();
    std::cout << "Starting CUDA training with " << n << " samples..." << std::endl;
    
    // allocate memory for the training data and gradients on GPU
    float *d_x, *d_y, *d_slope_grad, *d_intercept_grad;

    cudaMalloc(&d_x, n * sizeof(float)); 
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMalloc(&d_slope_grad, n * sizeof(float));
    cudaMalloc(&d_intercept_grad, n * sizeof(float));
    
    // copy data from cpu to gpu
    cudaMemcpy(d_x, x_train.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y_train.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    
    //initialise slope and parameter for first iteration
    slope_ = 0.0f;
    intercept_ = 0.0f;
    
    // training loop arbitrarily set to 1000 iterations
    int max_iterations = 1000;
    for (int iter = 0; iter < max_iterations; ++iter) {

        // define these for the kernel that will be used afterwards
        int blockSize = 256; //size of each block
        int numBlocks = (n + blockSize - 1) / blockSize; // number of blocks
        
        // using kernel to calculate gradients of slope and intercept
        calculateGradientsKernel<<<numBlocks, blockSize>>>(
            d_x, d_y, d_slope_grad, d_intercept_grad, slope_, intercept_, n
        );
        
        // load data back into cpu from gpu post kernel usage
        std::vector<float> slope_gradients(n);
        std::vector<float> intercept_gradients(n);
        cudaMemcpy(slope_gradients.data(), d_slope_grad, n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(intercept_gradients.data(), d_intercept_grad, n * sizeof(float), cudaMemcpyDeviceToHost);
        
        // sum gradients for total slope and intercept gradients (CPU Operation for Smaller datasets)
        float total_slope_grad = 0.0f;
        float total_intercept_grad = 0.0f;
        for (int i = 0; i < n; ++i) {
            total_slope_grad += slope_gradients[i];
            total_intercept_grad += intercept_gradients[i];
        }
        
        // accordingly update slope and intercept
        slope_ -= learning_rate_ * total_slope_grad / n;
        intercept_ -= learning_rate_ * total_intercept_grad / n;
    }
    
    // cleanup GPU memory (good practice)
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_slope_grad);
    cudaFree(d_intercept_grad);
    
    // set trained flag to true to enable predictions
    is_trained_ = true;
    std::cout << "CUDA training completed! Final slope=" << slope_ 
              << ", intercept=" << intercept_ << std::endl;
}

// prediction func
std::vector<float> LinearRegression::predict(const std::vector<float>& x_test) {
    if (!is_trained_) {
        throw std::runtime_error("Model must be trained before making predictions");
    }
    
    int n = x_test.size();
    
    // allocate GPU memory
    float *d_x, *d_predictions;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_predictions, n * sizeof(float));
    
    // copy test set to GPU
    cudaMemcpy(d_x, x_test.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    
    // define variables for kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    // use prediction kernel for computation of test set
    predictKernel<<<numBlocks, blockSize>>>(d_x, d_predictions, slope_, intercept_, n);
    
    // copy results back to cpu from gpu
    std::vector<float> predictions(n);
    cudaMemcpy(predictions.data(), d_predictions, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // cleanup (good practice for CUDA functions)
    cudaFree(d_x);
    cudaFree(d_predictions);
    
    return predictions;
}

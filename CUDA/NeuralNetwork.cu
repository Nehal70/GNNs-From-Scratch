#include "NeuralNetwork.h"
#include "Common/MatrixOps.cu"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>
#include <cmath>

namespace CUDA_ML {

// cuda kernels for activation functions used in neural networks
__global__ void reluKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void sigmoidKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = fmaxf(-500.0f, fminf(500.0f, input[idx]));
        output[idx] = 1.0f / (1.0f + expf(-x));
    }
}

__global__ void tanhKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

__global__ void leakyReluKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = (input[idx] > 0.0f) ? input[idx] : 0.01f * input[idx];
    }
}

__global__ void softmaxKernel(const float* input, float* output, int batch_size, int num_classes) {
    int batch_idx = blockIdx.x;
    int class_idx = threadIdx.x;
    
    if (batch_idx < batch_size && class_idx < num_classes) {
        // find maximum value for numerical stability
        float max_val = input[batch_idx * num_classes];
        for (int i = 1; i < num_classes; ++i) {
            max_val = fmaxf(max_val, input[batch_idx * num_classes + i]);
        }
        
        // calculate exponentials
        float sum = 0.0f;
        for (int i = 0; i < num_classes; ++i) {
            float exp_val = expf(input[batch_idx * num_classes + i] - max_val);
            output[batch_idx * num_classes + i] = exp_val;
            sum += exp_val;
        }
        
        // normalize
        output[batch_idx * num_classes + class_idx] /= sum;
    }
}

__global__ void reluDerivKernel(const float* activated_values, float* derivatives, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        derivatives[idx] = (activated_values[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

__global__ void sigmoidDerivKernel(const float* activated_values, float* derivatives, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        derivatives[idx] = activated_values[idx] * (1.0f - activated_values[idx]);
    }
}

__global__ void tanhDerivKernel(const float* activated_values, float* derivatives, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        derivatives[idx] = 1.0f - activated_values[idx] * activated_values[idx];
    }
}

__global__ void leakyReluDerivKernel(const float* activated_values, float* derivatives, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        derivatives[idx] = (activated_values[idx] > 0.0f) ? 1.0f : 0.01f;
    }
}

// helper functions for neural network activation functions
void NeuralNetwork::applyActivation(const float* input, float* output, const std::string& activation, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    if (activation == "relu") {
        reluKernel<<<numBlocks, blockSize>>>(input, output, size);
    } else if (activation == "sigmoid") {
        sigmoidKernel<<<numBlocks, blockSize>>>(input, output, size);
    } else if (activation == "tanh") {
        tanhKernel<<<numBlocks, blockSize>>>(input, output, size);
    } else if (activation == "leaky_relu") {
        leakyReluKernel<<<numBlocks, blockSize>>>(input, output, size);
    } else if (activation == "softmax") {
        // for softmax, we need batch size and number of classes
        // this is a simplified version for single sample per batch
        int batch_size = 1;
        int num_classes = size;
        softmaxKernel<<<batch_size, num_classes>>>(input, output, batch_size, num_classes);
    } else {
        // linear activation - just copy input to output
        cudaMemcpy(output, input, size * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    
    cudaDeviceSynchronize();
}

void NeuralNetwork::applyActivationDerivative(const float* activated_values, float* derivatives, const std::string& activation, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    if (activation == "relu") {
        reluDerivKernel<<<numBlocks, blockSize>>>(activated_values, derivatives, size);
    } else if (activation == "sigmoid") {
        sigmoidDerivKernel<<<numBlocks, blockSize>>>(activated_values, derivatives, size);
    } else if (activation == "tanh") {
        tanhDerivKernel<<<numBlocks, blockSize>>>(activated_values, derivatives, size);
    } else if (activation == "leaky_relu") {
        leakyReluDerivKernel<<<numBlocks, blockSize>>>(activated_values, derivatives, size);
    } else if (activation == "softmax") {
        // softmax derivative is more complex for proper implementation
        cudaMemset(derivatives, 1.0f, size * sizeof(float));
    } else {
        // linear activation derivative is all ones
        cudaMemset(derivatives, 1.0f, size * sizeof(float));
    }
    
    cudaDeviceSynchronize();
}

// constructor for the neural network that sets up layers, activations, and learning rate
NeuralNetwork::NeuralNetwork(const std::vector<int>& layers, 
                            const std::vector<std::string>& activations,
                            float learning_rate)
    : layers_(layers), activations_(activations), learning_rate_(learning_rate),
      is_trained_(false) {
    
    // make sure we have at least an input layer and output layer, otherwise this neural network wont work
    if (layers.size() < 2) {
        throw std::invalid_argument("At least input and output layers required");
    }
    // each layer except the input needs an activation function, so check that we have the right number
    if (activations.size() != layers.size() - 1) {
        throw std::invalid_argument("Activations size must equal layers.size()-1");
    }
    
    // go ahead and set up all the weights and biases for each layer
    initializeWeights();
}

NeuralNetwork::~NeuralNetwork() {
    // cleanup all the GPU memory we allocated for weights and biases so we dont leak memory
    for (auto& weight_matrix : d_weights_) {
        cudaFree(weight_matrix);
    }
    for (auto& bias_vector : d_biases_) {
        cudaFree(bias_vector);
    }
}

void NeuralNetwork::initializeWeights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (size_t i = 0; i < layers_.size() - 1; ++i) {
        int input_size = layers_[i];
        int output_size = layers_[i + 1];
        
        // get some GPU memory set aside for storing the weights between this layer and the next one
        float* d_weight;
        cudaMalloc(&d_weight, input_size * output_size * sizeof(float));
        
        // set up weights with random values following the xavier/he initialization method to avoid vanishing gradients
        std::vector<float> host_weights(input_size * output_size);
        float stddev = std::sqrt(2.0f / input_size);
        std::normal_distribution<float> dist(0.0f, stddev);
        
        for (auto& weight : host_weights) {
            weight = dist(gen);
        }
        
        // move the weights we just created on the cpu over to the gpu memory we allocated
        cudaMemcpy(d_weight, host_weights.data(), 
                  input_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
        
        d_weights_.push_back(d_weight);
        
        // biases start at zero since they represent the threshold for each neuron when it fires
        float* d_bias;
        cudaMalloc(&d_bias, output_size * sizeof(float));
        cudaMemset(d_bias, 0, output_size * sizeof(float));
        
        d_biases_.push_back(d_bias);
    }
}

void NeuralNetwork::forward(const float* d_input, int batch_size, 
                          std::vector<float*>& d_activations,
                          std::vector<float*>& d_z_values) {
    
    // clear out any previous activation values we stored so we start fresh
    d_activations.clear();
    d_z_values.clear();
    
    // make sure we have GPU memory allocated for storing all the intermediate neuron values through each layer
    for (size_t layer = 0; layer < layers_.size(); ++layer) {
        float* d_activation;
        cudaMalloc(&d_activation, batch_size * layers_[layer] * sizeof(float));
        d_activations.push_back(d_activation);
        
        if (layer < layers_.size() - 1) {
            float* d_z;
            cudaMalloc(&d_z, batch_size * layers_[layer + 1] * sizeof(float));
            d_z_values.push_back(d_z);
        }
    }
    
    // start the forward pass by copying our input data into the first layer of activations
    cudaMemcpy(d_activations[0], d_input, 
              batch_size * layers_[0] * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // now go through each layer and apply weights, biases, and activation functions
    for (size_t layer = 0; layer < layers_.size() - 1; ++layer) {
        int input_size = layers_[layer];
        int output_size = layers_[layer + 1];
        
        // do the basic linear algebra: multiply activations by weights and add bias
        MatrixOps::matmulAddBias(d_activations[layer], d_weights_[layer], d_biases_[layer],
                                d_z_values[layer], batch_size, input_size, output_size);
        
        // apply whatever activation function we chose for this layer to make things nonlinear
        applyActivation(d_z_values[layer], d_activations[layer + 1],
                      activations_[layer], batch_size * output_size);
    }
}

void NeuralNetwork::backward(const float* d_input, const float* d_y_true,
                           const std::vector<float*>& d_activations,
                           const std::vector<float*>& d_z_values,
                           int batch_size) {
    
    // figure out how wrong our final prediction was compared to what we wanted
    float* d_output_error;
    cudaMalloc(&d_output_error, batch_size * layers_.back() * sizeof(float));
    
    MatrixOps::subtract(d_activations.back(), d_y_true, d_output_error,
                      batch_size * layers_.back());
    
    // now work backwards through each layer to figure out how much each weight and bias contributed to the error
    for (int layer = static_cast<int>(layers_.size()) - 2; layer >= 0; --layer) {
        int input_size = layers_[layer];
        int output_size = layers_[layer + 1];
        
        // figure out how much we need to adjust all the weights for this layer to reduce the error
        float* d_weight_grad;
        cudaMalloc(&d_weight_grad, input_size * output_size * sizeof(float));
        
        MatrixOps::matmulTranspose(d_activations[layer], d_output_error, d_weight_grad,
                                 batch_size, input_size, output_size);
        
        // similarly figure out how much to adjust the biases for this layer
        float* d_bias_grad;
        cudaMalloc(&d_bias_grad, output_size * sizeof(float));
        
        MatrixOps::sumRows(d_output_error, d_bias_grad, batch_size, output_size);
        
        // actually update the weights and biases using our calculated gradients and the learning rate
        MatrixOps::updateWeights(d_weights_[layer], d_weight_grad, 
                               learning_rate_ / batch_size, input_size * output_size);
        MatrixOps::updateBiases(d_biases_[layer], d_bias_grad,
                              learning_rate_ / batch_size, output_size);
        
        // if this isnt the first layer, figure out how much error to pass back to the previous layer
        if (layer > 0) {
            float* d_prev_error;
            cudaMalloc(&d_prev_error, batch_size * input_size * sizeof(float));
            
            MatrixOps::matmulTranspose(d_output_error, d_weights_[layer], d_prev_error,
                                     batch_size, output_size, input_size);
            
            // apply the derivative the activation function to adjust the error appropriately
            applyActivationDerivative(d_activations[layer], d_prev_error,
                                   activations_[layer - 1],
                                   batch_size * input_size);
            
            cudaFree(d_output_error);
            d_output_error = d_prev_error;
        }
        
        // free up these gradient arrays since we dont need them anymore
        cudaFree(d_weight_grad);
        cudaFree(d_bias_grad);
    }
    
    // cleanup the final error array
    cudaFree(d_output_error);
}

void NeuralNetwork::train(const std::vector<std::vector<float>>& X_train,
                        const std::vector<std::vector<float>>& y_train,
                        int epochs, bool verbose) {
    
    if (X_train.size() != y_train.size()) {
        throw std::invalid_argument("Training data sizes must match");
    }
    
    int batch_size = X_train.size();
    int input_size = layers_[0];
    int output_size = layers_.back();
    
    // convert our 2d training data into flat 1d arrays since the GPU kernels prefer working with flat data
    std::vector<float> flat_X(batch_size * input_size);
    std::vector<float> flat_y(batch_size * output_size);
    
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            flat_X[i * input_size + j] = X_train[i][j];
        }
        for (int j = 0; j < output_size; ++j) {
            flat_y[i * output_size + j] = y_train[i][j];
        }
    }
    
    // make space on the GPU for our training data
    float *d_X, *d_y;
    cudaMalloc(&d_X, batch_size * input_size * sizeof(float));
    cudaMalloc(&d_y, batch_size * output_size * sizeof(float));
    
    // copy the training data from cpu memory to gpu memory so we can work with it on the gpu
    cudaMemcpy(d_X, flat_X.data(), batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, flat_y.data(), batch_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // now run through all the training epochs we specified
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<float*> d_activations, d_z_values;
        
        // send the input data forward through all layers to get predictions
        forward(d_X, batch_size, d_activations, d_z_values);
        
        // work backwards from the predictions to update all weights and biases
        backward(d_X, d_y, d_activations, d_z_values, batch_size);
        
        // optionally print out the loss every 10 epochs so we can see how training is going
        if (verbose && (epoch % 10 == 0 || epoch == epochs - 1)) {
            float loss = calculateLoss(d_y, d_activations.back(), batch_size);
            std::cout << "Epoch " << epoch << "/" << epochs << ", Loss: " << loss << std::endl;
        }
        
        // free up the memory we used for intermediate calculations during this epoch
        for (auto& activation : d_activations) {
            cudaFree(activation);
        }
        for (auto& z_value : d_z_values) {
            cudaFree(z_value);
        }
    }
    
    // clean up the gpu memory we allocated for the training data
    cudaFree(d_X);
    cudaFree(d_y);
    
    // mark the network as trained so we can use it to make predictions
    is_trained_ = true;
}

std::vector<std::vector<float>> NeuralNetwork::predict(const std::vector<std::vector<float>>& X_test) {
    if (!is_trained_) {
        throw std::runtime_error("Model must be trained before making predictions");
    }
    
    int batch_size = X_test.size();
    int input_size = layers_[0];
    int output_size = layers_.back();
    
    // flatten the test data just like we did for training data since we need flat arrays for the gpu
    std::vector<float> flat_X(batch_size * input_size);
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            flat_X[i * input_size + j] = X_test[i][j];
        }
    }
    
    // allocate gpu memory for the test data so we can run it through the network
    float *d_X;
    cudaMalloc(&d_X, batch_size * input_size * sizeof(float));
    cudaMemcpy(d_X, flat_X.data(), batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // run the test data through the network to get our predictions
    std::vector<float*> d_activations, d_z_values;
    forward(d_X, batch_size, d_activations, d_z_values);
    
    // copy the final predictions back from gpu to cpu memory so we can return them
    std::vector<float> flat_predictions(batch_size * output_size);
    cudaMemcpy(flat_predictions.data(), d_activations.back(),
              batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // convert the flat predictions back to a 2d structure that matches our input format
    std::vector<std::vector<float>> predictions(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        predictions[i].resize(output_size);
        for (int j = 0; j < output_size; ++j) {
            predictions[i][j] = flat_predictions[i * output_size + j];
        }
    }
    
    // clean up all the gpu memory we allocated during this prediction
    cudaFree(d_X);
    for (auto& activation : d_activations) {
        cudaFree(activation);
    }
    for (auto& z_value : d_z_values) {
        cudaFree(z_value);
    }
    
    return predictions;
}

float NeuralNetwork::calculateLoss(const float* d_y_true, const float* d_y_pred, int batch_size) {
    float* d_loss;
    cudaMalloc(&d_loss, batch_size * sizeof(float));
    
    // calculate how much each prediction differs from the true value squared
    MatrixOps::squaredError(d_y_true, d_y_pred, d_loss, batch_size * layers_.back());
    
    // sum up all the losses and divide by number of samples to get the average loss
    float loss;
    MatrixOps::sumReduce(d_loss, &loss, batch_size);
    loss /= batch_size;
    
    // cleanup the temporary memory we used for loss calculation
    cudaFree(d_loss);
    return loss;
}

} // namespace CUDA_ML


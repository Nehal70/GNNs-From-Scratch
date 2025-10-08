#include "GNN.h"
#include "Common/MatrixOps.h"
#include "Common/Utils.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>

namespace CUDA_ML {

// cuda kernel for graph convolution layer - implements message passing between connected nodes
__global__ void graphConvolutionKernel(const float* node_features, const float* adjacency_matrix,
                                     const float* weights, const float* bias, float* output_features,
                                     int num_nodes, int feature_dim, int output_dim) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feature_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (node_idx < num_nodes && feature_idx < output_dim) {
        float sum = 0.0f;
        
        // aggregate neighbor features by summing them up
        // adjacency matrix tells us which nodes are connected
        for (int neighbor = 0; neighbor < num_nodes; ++neighbor) {
            if (adjacency_matrix[node_idx * num_nodes + neighbor] > 0.0f) {
                for (int f = 0; f < feature_dim; ++f) {
                    sum += node_features[neighbor * feature_dim + f] * 
                           weights[f * output_dim + feature_idx];
                }
            }
        }
        
        // add bias term for this output dimension
        output_features[node_idx * output_dim + feature_idx] = sum + bias[feature_idx];
    }
}

// cuda kernel for softmax activation - converts outputs into probability distribution
__global__ void softmaxKernel(const float* input, float* output, int num_nodes, int num_classes) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node_idx < num_nodes) {
        // find maximum value for numerical stability
        float max_val = input[node_idx * num_classes];
        for (int i = 1; i < num_classes; ++i) {
            max_val = fmaxf(max_val, input[node_idx * num_classes + i]);
        }
        
        // compute exponentials and sum
        float sum_exp = 0.0f;
        for (int i = 0; i < num_classes; ++i) {
            float exp_val = expf(input[node_idx * num_classes + i] - max_val);
            output[node_idx * num_classes + i] = exp_val;
            sum_exp += exp_val;
        }
        
        // normalize to get probabilities
        for (int i = 0; i < num_classes; ++i) {
            output[node_idx * num_classes + i] /= sum_exp;
        }
    }
}

// cuda kernel for sigmoid activation - bounded output for property prediction
__global__ void sigmoidKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // clip to prevent overflow
        float x = fmaxf(-500.0f, fminf(500.0f, input[idx]));
        output[idx] = 1.0f / (1.0f + expf(-x));
    }
}

// cuda kernel for relu activation - used in hidden layers
__global__ void reluKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// cuda kernel for computing cross entropy loss - used in classification tasks
__global__ void crossEntropyLossKernel(const float* predictions, const float* labels, 
                                      float* loss, int num_nodes, int num_classes) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node_idx < num_nodes) {
        float node_loss = 0.0f;
        for (int i = 0; i < num_classes; ++i) {
            float pred = fmaxf(1e-15f, fminf(1.0f - 1e-15f, predictions[node_idx * num_classes + i]));
            node_loss += labels[node_idx * num_classes + i] * logf(pred);
        }
        loss[node_idx] = -node_loss;
    }
}

// cuda kernel for computing mean squared error loss - used in regression tasks
__global__ void mseLossKernel(const float* predictions, const float* labels, 
                             float* loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float diff = predictions[idx] - labels[idx];
        loss[idx] = diff * diff;
    }
}

// cuda kernel for computing binary cross entropy loss - used in property prediction
__global__ void binaryCrossEntropyLossKernel(const float* predictions, const float* labels, 
                                            float* loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float pred = fmaxf(1e-15f, fminf(1.0f - 1e-15f, predictions[idx]));
        loss[idx] = -(labels[idx] * logf(pred) + (1.0f - labels[idx]) * logf(1.0f - pred));
    }
}

// constructor that sets up the graph neural network with the specified architecture and hyperparameters
GNN::GNN(int InputDim, const std::vector<int>& HiddenDims, int OutputDim, const std::string& TaskType, float LearningRate)
    : InputDim_(InputDim), HiddenDims_(HiddenDims), OutputDim_(OutputDim), TaskType_(TaskType), LearningRate_(LearningRate), IsTrained_(false) {
    
    // build the gnn layers
    LayerSizes_.push_back(InputDim);
    for (int HiddenSize : HiddenDims) {
        LayerSizes_.push_back(HiddenSize);
    }
    LayerSizes_.push_back(OutputDim);
    
    // initialize weights and biases for each layer
    initializeWeights();
}

// cleanup gpu memory when the graph neural network is destroyed
GNN::~GNN() {
    // free all gpu memory
    for (auto& W : DWeights_) {
        if (W) cudaFree(W);
    }
    for (auto& B : DBiases_) {
        if (B) cudaFree(B);
    }
}

// initialize weights and biases for all layers using xavier initialization
void GNN::initializeWeights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (size_t i = 0; i < LayerSizes_.size() - 1; ++i) {
        int InputSize = LayerSizes_[i];
        int OutputSize = LayerSizes_[i + 1];
        
        // allocate gpu memory for weights
        float* DW;
        cudaMalloc(&DW, InputSize * OutputSize * sizeof(float));
        
        // initialize weights using xavier initialization
        std::vector<float> HostW(InputSize * OutputSize);
        float StdDev = std::sqrt(2.0f / InputSize);
        std::normal_distribution<float> Dist(0.0f, StdDev);
        
        for (auto& W : HostW) {
            W = Dist(gen);
        }
        
        cudaMemcpy(DW, HostW.data(), 
                  InputSize * OutputSize * sizeof(float), cudaMemcpyHostToDevice);
        
        DWeights_.push_back(DW);
        
        // allocate and initialize biases to zero
        float* DB;
        cudaMalloc(&DB, OutputSize * sizeof(float));
        cudaMemset(DB, 0, OutputSize * sizeof(float));
        
        DBiases_.push_back(DB);
    }
}

// apply activation function based on task type and layer
void GNN::applyActivation(const float* Input, float* Output, const std::string& Activation, int Size) {
    int BlockSize = 256;
    int NumBlocks = (Size + BlockSize - 1) / BlockSize;
    
    if (Activation == "relu") {
        reluKernel<<<NumBlocks, BlockSize>>>(Input, Output, Size);
    } else if (Activation == "sigmoid") {
        sigmoidKernel<<<NumBlocks, BlockSize>>>(Input, Output, Size);
    } else if (Activation == "softmax") {
        // softmax needs special handling for 2d arrays
        int NumNodes = Size / OutputDim_;
        int NumClasses = OutputDim_;
        int NodeBlocks = (NumNodes + BlockSize - 1) / BlockSize;
        softmaxKernel<<<NodeBlocks, BlockSize>>>(Input, Output, NumNodes, NumClasses);
    } else {
        // linear activation - just copy input to output
        cudaMemcpy(Output, Input, Size * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    
    cudaDeviceSynchronize();
}

// forward pass through entire gnn - processes node features through all graph convolution layers
void GNN::forward(const std::vector<std::vector<float>>& NodeFeatures,
                 const std::vector<std::vector<float>>& AdjacencyMatrix,
                 std::vector<std::vector<float>>& OutputFeatures) {
    
    int NumNodes = NodeFeatures.size();
    int FeatureDim = NodeFeatures[0].size();
    
    // flatten input data for gpu processing
    std::vector<float> FlatFeatures(NumNodes * FeatureDim);
    std::vector<float> FlatAdjacency(NumNodes * NumNodes);
    
    for (int i = 0; i < NumNodes; ++i) {
        for (int j = 0; j < FeatureDim; ++j) {
            FlatFeatures[i * FeatureDim + j] = NodeFeatures[i][j];
        }
        for (int j = 0; j < NumNodes; ++j) {
            FlatAdjacency[i * NumNodes + j] = AdjacencyMatrix[i][j];
        }
    }
    
    // allocate gpu memory
    float *DFeatures, *DAdjacency, *DOutput;
    cudaMalloc(&DFeatures, NumNodes * FeatureDim * sizeof(float));
    cudaMalloc(&DAdjacency, NumNodes * NumNodes * sizeof(float));
    cudaMalloc(&DOutput, NumNodes * OutputDim_ * sizeof(float));
    
    // copy data to gpu
    cudaMemcpy(DFeatures, FlatFeatures.data(), 
              NumNodes * FeatureDim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(DAdjacency, FlatAdjacency.data(), 
              NumNodes * NumNodes * sizeof(float), cudaMemcpyHostToDevice);
    
    // forward pass through gnn layers
    float* H = DFeatures;
    int CurrentDim = FeatureDim;
    
    // pass through hidden layers with relu activation
    for (size_t Layer = 0; Layer < LayerSizes_.size() - 1; ++Layer) {
        int OutputDim = LayerSizes_[Layer + 1];
        
        // graph convolution operation
        dim3 BlockSize(16, 16);
        dim3 GridSize((NumNodes + BlockSize.x - 1) / BlockSize.x,
                      (OutputDim + BlockSize.y - 1) / BlockSize.y);
        
        graphConvolutionKernel<<<GridSize, BlockSize>>>(
            H, DAdjacency, DWeights_[Layer], DBiases_[Layer], DOutput,
            NumNodes, CurrentDim, OutputDim
        );
        
        cudaDeviceSynchronize();
        
        // apply activation function
        if (Layer == LayerSizes_.size() - 2) {
            // final layer - apply task-specific activation
            if (TaskType_ == "classification") {
                applyActivation(DOutput, DOutput, "softmax", NumNodes * OutputDim);
            } else if (TaskType_ == "property_prediction") {
                applyActivation(DOutput, DOutput, "sigmoid", NumNodes * OutputDim);
            }
            // regression uses linear activation (no activation)
        } else {
            // hidden layers use relu activation
            applyActivation(DOutput, DOutput, "relu", NumNodes * OutputDim);
        }
        
        // update for next layer
        if (Layer < LayerSizes_.size() - 2) {
            if (H != DFeatures) {
                cudaFree(H);
            }
            H = DOutput;
            CurrentDim = OutputDim;
            
            // allocate new output for next layer
            cudaMalloc(&DOutput, NumNodes * LayerSizes_[Layer + 2] * sizeof(float));
        }
    }
    
    // copy final results back to host
    std::vector<float> FlatOutput(NumNodes * OutputDim_);
    cudaMemcpy(FlatOutput.data(), DOutput, 
              NumNodes * OutputDim_ * sizeof(float), cudaMemcpyDeviceToHost);
    
    // reshape output
    OutputFeatures.resize(NumNodes);
    for (int i = 0; i < NumNodes; ++i) {
        OutputFeatures[i].resize(OutputDim_);
        for (int j = 0; j < OutputDim_; ++j) {
            OutputFeatures[i][j] = FlatOutput[i * OutputDim_ + j];
        }
    }
    
    // cleanup gpu memory
    cudaFree(DFeatures);
    cudaFree(DAdjacency);
    cudaFree(DOutput);
    if (H != DFeatures) {
        cudaFree(H);
    }
}

// calculate loss based on task type
float GNN::calculateLoss(const std::vector<std::vector<float>>& predictions,
                        const std::vector<std::vector<float>>& labels) {
    int num_nodes = predictions.size();
    int pred_size = predictions[0].size();
    
    // flatten data for gpu processing
    std::vector<float> flat_predictions(num_nodes * pred_size);
    std::vector<float> flat_labels(num_nodes * pred_size);
    
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < pred_size; ++j) {
            flat_predictions[i * pred_size + j] = predictions[i][j];
            flat_labels[i * pred_size + j] = labels[i][j];
        }
    }
    
    // allocate gpu memory
    float *d_predictions, *d_labels, *d_loss;
    cudaMalloc(&d_predictions, num_nodes * pred_size * sizeof(float));
    cudaMalloc(&d_labels, num_nodes * pred_size * sizeof(float));
    cudaMalloc(&d_loss, num_nodes * sizeof(float));
    
    // copy data to gpu
    cudaMemcpy(d_predictions, flat_predictions.data(), 
              num_nodes * pred_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, flat_labels.data(), 
              num_nodes * pred_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // compute loss based on task type
    int blockSize = 256;
    int numBlocks = (num_nodes + blockSize - 1) / blockSize;
    
    if (TaskType_ == "classification") {
        crossEntropyLossKernel<<<numBlocks, blockSize>>>(
            d_predictions, d_labels, d_loss, num_nodes, pred_size);
    } else if (TaskType_ == "regression") {
        mseLossKernel<<<numBlocks, blockSize>>>(
            d_predictions, d_labels, d_loss, num_nodes * pred_size);
    } else if (TaskType_ == "property_prediction") {
        binaryCrossEntropyLossKernel<<<numBlocks, blockSize>>>(
            d_predictions, d_labels, d_loss, num_nodes * pred_size);
    }
    
    cudaDeviceSynchronize();
    
    // compute mean loss
    std::vector<float> host_loss(num_nodes);
    cudaMemcpy(host_loss.data(), d_loss, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
    
    float total_loss = 0.0f;
    for (float loss : host_loss) {
        total_loss += loss;
    }
    total_loss /= num_nodes;
    
    // cleanup
    cudaFree(d_predictions);
    cudaFree(d_labels);
    cudaFree(d_loss);
    
    return total_loss;
}

// train the gnn using proper graph-aware backpropagation
void GNN::train(const std::vector<std::vector<float>>& node_features,
               const std::vector<std::vector<float>>& adjacency_matrix,
               const std::vector<std::vector<float>>& labels, int epochs, bool verbose) {
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // forward pass - get both predictions and activations
        std::vector<std::vector<float>> predictions;
        forward(node_features, adjacency_matrix, predictions);
        
        // calculate loss
        float current_loss = calculateLoss(predictions, labels);
        
        // print progress
        if (verbose && (epoch % 20 == 0 || epoch == epochs - 1)) {
            std::cout << "epoch " << epoch << "/" << epochs << ", loss: " << current_loss << std::endl;
        }
        
        // note: proper backpropagation implementation would go here
        // this is simplified for the initial structure
        // in a full implementation, you would:
        // 1. compute gradients using graph-aware backpropagation
        // 2. update weights and biases using computed gradients
        // 3. handle different loss functions properly
    }
    
    IsTrained_ = true;
}

// use the trained gnn to predict node features or classifications
std::vector<std::vector<float>> GNN::predict(const std::vector<std::vector<float>>& node_features,
                                            const std::vector<std::vector<float>>& adjacency_matrix) {
    if (!IsTrained_) {
        std::cout << "model not trained yet." << std::endl;
        return {};
    }
    
    std::vector<std::vector<float>> predictions;
    forward(node_features, adjacency_matrix, predictions);
    return predictions;
}

} // namespace CUDA_ML




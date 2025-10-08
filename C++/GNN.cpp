#include "GNN.h"
#include <iostream>
#include <iomanip>
#include <chrono>

namespace CPP_ML {

// constructor that initializes weights and bias for this layer
GraphConvolutionLayer::GraphConvolutionLayer(int InputDim, int OutputDim) {
    // initialize weights using xavier initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    float StdDev = std::sqrt(2.0f / InputDim);
    std::normal_distribution<float> Dist(0.0f, StdDev);
    
    Weights_.resize(InputDim);
    for (int i = 0; i < InputDim; ++i) {
        Weights_[i].resize(OutputDim);
        for (int j = 0; j < OutputDim; ++j) {
            Weights_[i][j] = Dist(gen);
        }
    }
    
    // initialize biases to zero
    Biases_.resize(OutputDim, 0.0f);
}

// forward pass through graph convolution layer
std::vector<std::vector<float>> GraphConvolutionLayer::forward(
    const std::vector<std::vector<float>>& NodeFeatures,
    const std::vector<std::vector<float>>& AdjacencyMatrix) {
    
    int NumNodes = NodeFeatures.size();
    int FeatureDim = NodeFeatures[0].size();
    int OutputDim = Biases_.size();
    
    std::vector<std::vector<float>> Output(NumNodes, std::vector<float>(OutputDim, 0.0f));
    
    // step 1: aggregate neighbor features by summing them up
    // adjacency matrix tells us which nodes are connected
    for (int node = 0; node < NumNodes; ++node) {
        for (int neighbor = 0; neighbor < NumNodes; ++neighbor) {
            if (AdjacencyMatrix[node][neighbor] > 0.0f) {
                for (int out_dim = 0; out_dim < OutputDim; ++out_dim) {
                    for (int feat = 0; feat < FeatureDim; ++feat) {
                        Output[node][out_dim] += NodeFeatures[neighbor][feat] * Weights_[feat][out_dim];
                    }
                }
            }
        }
    }
    
    // step 2: add bias term for each output dimension
    for (int node = 0; node < NumNodes; ++node) {
        for (int out_dim = 0; out_dim < OutputDim; ++out_dim) {
            Output[node][out_dim] += Biases_[out_dim];
        }
    }
    
    return Output;
}

// constructor that sets up the graph neural network with the specified architecture and hyperparameters
GNN::GNN(int InputDim, const std::vector<int>& HiddenDims, int OutputDim, 
         const std::string& TaskType, float LearningRate)
    : InputDim_(InputDim), HiddenDims_(HiddenDims), OutputDim_(OutputDim),
      TaskType_(TaskType), LearningRate_(LearningRate), IsTrained_(false) {
    
    // build the gnn layers
    LayerSizes_.push_back(InputDim);
    for (int HiddenSize : HiddenDims) {
        LayerSizes_.push_back(HiddenSize);
    }
    LayerSizes_.push_back(OutputDim);
    
    // initialize graph convolution layers
    int CurrentDim = InputDim;
    for (int HiddenSize : HiddenDims) {
        GNNLayers_.emplace_back(CurrentDim, HiddenSize);
        CurrentDim = HiddenSize;
    }
    
    // initialize final layer
    FinalLayer_ = GraphConvolutionLayer(CurrentDim, OutputDim);
}

// apply activation function based on task type and layer
std::vector<std::vector<float>> GNN::applyActivation(const std::vector<std::vector<float>>& Input, 
                                                   const std::string& Activation) {
    int NumNodes = Input.size();
    int FeatureDim = Input[0].size();
    std::vector<std::vector<float>> Output = Input;
    
    if (Activation == "relu") {
        for (int i = 0; i < NumNodes; ++i) {
            for (int j = 0; j < FeatureDim; ++j) {
                Output[i][j] = std::max(0.0f, Input[i][j]);
            }
        }
    } else if (Activation == "sigmoid") {
        for (int i = 0; i < NumNodes; ++i) {
            for (int j = 0; j < FeatureDim; ++j) {
                float X = std::max(-500.0f, std::min(500.0f, Input[i][j]));
                Output[i][j] = 1.0f / (1.0f + std::exp(-X));
            }
        }
    } else if (Activation == "softmax") {
        for (int i = 0; i < NumNodes; ++i) {
            // find maximum for numerical stability
            float MaxVal = *std::max_element(Input[i].begin(), Input[i].end());
            
            // compute exponentials and sum
            float SumExp = 0.0f;
            for (int j = 0; j < FeatureDim; ++j) {
                Output[i][j] = std::exp(Input[i][j] - MaxVal);
                SumExp += Output[i][j];
            }
            
            // normalize to get probabilities
            for (int j = 0; j < FeatureDim; ++j) {
                Output[i][j] /= SumExp;
            }
        }
    }
    // linear activation - no change needed
    
    return Output;
}

// forward pass through entire gnn - processes node features through all graph convolution layers
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>> 
GNN::forward(const std::vector<std::vector<float>>& NodeFeatures,
             const std::vector<std::vector<float>>& AdjacencyMatrix) {
    
    std::vector<std::vector<std::vector<float>>> Activations;
    std::vector<std::vector<float>> H = NodeFeatures;
    Activations.push_back(H);
    
    // pass through hidden layers with relu activation
    for (auto& ConvLayer : GNNLayers_) {
        H = ConvLayer.forward(H, AdjacencyMatrix);
        H = applyActivation(H, "relu");
        Activations.push_back(H);
    }
    
    // final output layer
    std::vector<std::vector<float>> FinalOutput = FinalLayer_.forward(H, AdjacencyMatrix);
    
    // apply final activation based on task type
    if (TaskType_ == "classification") {
        FinalOutput = applyActivation(FinalOutput, "softmax");
    } else if (TaskType_ == "property_prediction") {
        FinalOutput = applyActivation(FinalOutput, "sigmoid");
    }
    // regression uses linear activation (no activation)
    
    Activations.push_back(FinalOutput);
    return std::make_pair(FinalOutput, Activations);
}

// calculate loss based on task type
float GNN::calculateLoss(const std::vector<std::vector<float>>& Predictions,
                        const std::vector<std::vector<float>>& Labels) {
    int NumNodes = Predictions.size();
    int PredDim = Predictions[0].size();
    
    if (TaskType_ == "classification") {
        // cross-entropy loss for classification tasks
        float TotalLoss = 0.0f;
        for (int i = 0; i < NumNodes; ++i) {
            for (int j = 0; j < PredDim; ++j) {
                float Pred = std::max(1e-15f, std::min(1.0f - 1e-15f, Predictions[i][j]));
                TotalLoss += Labels[i][j] * std::log(Pred);
            }
        }
        return -TotalLoss / NumNodes;
    } else if (TaskType_ == "regression") {
        // mean squared error loss for regression tasks
        float TotalLoss = 0.0f;
        for (int i = 0; i < NumNodes; ++i) {
            for (int j = 0; j < PredDim; ++j) {
                float Diff = Labels[i][j] - Predictions[i][j];
                TotalLoss += Diff * Diff;
            }
        }
        return TotalLoss / NumNodes;
    } else if (TaskType_ == "property_prediction") {
        // binary cross entropy for property prediction tasks
        float TotalLoss = 0.0f;
        for (int i = 0; i < NumNodes; ++i) {
            for (int j = 0; j < PredDim; ++j) {
                float Pred = std::max(1e-15f, std::min(1.0f - 1e-15f, Predictions[i][j]));
                TotalLoss += Labels[i][j] * std::log(Pred) + (1.0f - Labels[i][j]) * std::log(1.0f - Pred);
            }
        }
        return -TotalLoss / NumNodes;
    }
    
    return 0.0f;
}

// compute gradients properly for gnn using graph-aware backpropagation
std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<float>>> 
GNN::computeGraphGradients(const std::vector<std::vector<float>>& NodeFeatures,
                           const std::vector<std::vector<float>>& AdjacencyMatrix,
                           const std::vector<std::vector<float>>& Labels,
                           const std::vector<std::vector<std::vector<float>>>& Activations) {
    
    int NumNodes = NodeFeatures.size();
    int FeatureDim = NodeFeatures[0].size();
    
    // compute output layer gradient
    std::vector<std::vector<float>> OutputGrad = Activations.back();
    for (int i = 0; i < NumNodes; ++i) {
        for (int j = 0; j < OutputDim_; ++j) {
            OutputGrad[i][j] -= Labels[i][j];
        }
    }
    
    // store gradients for each layer
    std::vector<std::vector<std::vector<float>>> WeightGrads;
    std::vector<std::vector<float>> BiasGrads;
    
    // backprop through final layer
    std::vector<std::vector<float>> FinalLayerGradW(LayerSizes_[LayerSizes_.size() - 2], 
                                                   std::vector<float>(OutputDim_, 0.0f));
    std::vector<float> FinalLayerGradB(OutputDim_, 0.0f);
    
    // compute weight gradients for final layer
    for (int i = 0; i < LayerSizes_[LayerSizes_.size() - 2]; ++i) {
        for (int j = 0; j < OutputDim_; ++j) {
            for (int n = 0; n < NumNodes; ++n) {
                FinalLayerGradW[i][j] += Activations[Activations.size() - 2][n][i] * OutputGrad[n][j];
            }
        }
    }
    
    // compute bias gradients for final layer
    for (int j = 0; j < OutputDim_; ++j) {
        for (int n = 0; n < NumNodes; ++n) {
            FinalLayerGradB[j] += OutputGrad[n][j];
        }
        FinalLayerGradB[j] /= NumNodes;
    }
    
    WeightGrads.push_back(FinalLayerGradW);
    BiasGrads.push_back(FinalLayerGradB);
    
    // compute gradient flowing back through final layer
    std::vector<std::vector<float>> CurrentGrad = backpropGraphConv(OutputGrad, AdjacencyMatrix, 
                                                                    FinalLayer_.getWeights());
    
    // backprop through hidden layers with graph structure
    for (int i = GNNLayers_.size() - 1; i >= 0; --i) {
        // apply activation derivative for current layer
        CurrentGrad = reluDerivative(CurrentGrad);
        
        // compute gradients for current layer weights
        std::vector<std::vector<float>> LayerGradW(LayerSizes_[i], 
                                                  std::vector<float>(LayerSizes_[i + 1], 0.0f));
        std::vector<float> LayerGradB(LayerSizes_[i + 1], 0.0f);
        
        // weight gradient: dW = input.T @ gradient
        for (int k = 0; k < LayerSizes_[i]; ++k) {
            for (int l = 0; l < LayerSizes_[i + 1]; ++l) {
                for (int n = 0; n < NumNodes; ++n) {
                    LayerGradW[k][l] += Activations[i][n][k] * CurrentGrad[n][l];
                }
            }
        }
        
        // bias gradient: db = mean(gradient)
        for (int l = 0; l < LayerSizes_[i + 1]; ++l) {
            for (int n = 0; n < NumNodes; ++n) {
                LayerGradB[l] += CurrentGrad[n][l];
            }
            LayerGradB[l] /= NumNodes;
        }
        
        WeightGrads.insert(WeightGrads.begin(), LayerGradW);
        BiasGrads.insert(BiasGrads.begin(), LayerGradB);
        
        // propagate gradient backward through graph convolution
        if (i > 0) {
            CurrentGrad = backpropGraphConv(CurrentGrad, AdjacencyMatrix, GNNLayers_[i].getWeights());
        }
    }
    
    return std::make_pair(WeightGrads, BiasGrads);
}

// backpropagate gradient through graph convolution operation
std::vector<std::vector<float>> GNN::backpropGraphConv(const std::vector<std::vector<float>>& GradOutput,
                                                      const std::vector<std::vector<float>>& AdjacencyMatrix,
                                                      const std::vector<std::vector<float>>& WeightMatrix) {
    
    int NumNodes = GradOutput.size();
    int OutputDim = GradOutput[0].size();
    int InputDim = WeightMatrix.size();
    
    std::vector<std::vector<float>> GradInput(NumNodes, std::vector<float>(InputDim, 0.0f));
    
    // gradient flows back through the linear transformation
    for (int node = 0; node < NumNodes; ++node) {
        for (int feat = 0; feat < InputDim; ++feat) {
            for (int out_dim = 0; out_dim < OutputDim; ++out_dim) {
                GradInput[node][feat] += GradOutput[node][out_dim] * WeightMatrix[feat][out_dim];
            }
        }
    }
    
    // gradient flows back through the adjacency matrix multiplication
    // this is the graph-aware part: gradients propagate through connections
    std::vector<std::vector<float>> FinalGradInput(NumNodes, std::vector<float>(InputDim, 0.0f));
    for (int node = 0; node < NumNodes; ++node) {
        for (int neighbor = 0; neighbor < NumNodes; ++neighbor) {
            if (AdjacencyMatrix[node][neighbor] > 0.0f) {
                for (int feat = 0; feat < InputDim; ++feat) {
                    FinalGradInput[node][feat] += GradInput[neighbor][feat];
                }
            }
        }
    }
    
    return FinalGradInput;
}

// compute derivative of relu activation function
std::vector<std::vector<float>> GNN::reluDerivative(const std::vector<std::vector<float>>& Input) {
    std::vector<std::vector<float>> Output = Input;
    for (size_t i = 0; i < Input.size(); ++i) {
        for (size_t j = 0; j < Input[i].size(); ++j) {
            Output[i][j] = (Input[i][j] > 0.0f) ? 1.0f : 0.0f;
        }
    }
    return Output;
}

// train the gnn using sophisticated graph-aware backpropagation
void GNN::train(const std::vector<std::vector<float>>& NodeFeatures,
                const std::vector<std::vector<float>>& AdjacencyMatrix,
                const std::vector<std::vector<float>>& Labels, int Epochs, bool Verbose) {
    
    for (int epoch = 0; epoch < Epochs; ++epoch) {
        // forward pass - get both predictions and activations
        auto [Predictions, Activations] = forward(NodeFeatures, AdjacencyMatrix);
        
        // calculate loss
        float CurrentLoss = calculateLoss(Predictions, Labels);
        
        // compute gradients using graph-aware backpropagation
        auto [WeightGrads, BiasGrads] = computeGraphGradients(NodeFeatures, AdjacencyMatrix, Labels, Activations);
        
        // update weights and biases using computed gradients
        for (size_t i = 0; i < GNNLayers_.size(); ++i) {
            auto CurrentWeights = GNNLayers_[i].getWeights();
            auto CurrentBiases = GNNLayers_[i].getBiases();
            
            // update weights
            for (size_t j = 0; j < CurrentWeights.size(); ++j) {
                for (size_t k = 0; k < CurrentWeights[j].size(); ++k) {
                    CurrentWeights[j][k] -= LearningRate_ * WeightGrads[i][j][k];
                }
            }
            
            // update biases
            for (size_t j = 0; j < CurrentBiases.size(); ++j) {
                CurrentBiases[j] -= LearningRate_ * BiasGrads[i][j];
            }
            
            GNNLayers_[i].setWeights(CurrentWeights);
            GNNLayers_[i].setBiases(CurrentBiases);
        }
        
        // update final layer
        auto FinalWeights = FinalLayer_.getWeights();
        auto FinalBiases = FinalLayer_.getBiases();
        
        for (size_t j = 0; j < FinalWeights.size(); ++j) {
            for (size_t k = 0; k < FinalWeights[j].size(); ++k) {
                FinalWeights[j][k] -= LearningRate_ * WeightGrads.back()[j][k];
            }
        }
        
        for (size_t j = 0; j < FinalBiases.size(); ++j) {
            FinalBiases[j] -= LearningRate_ * BiasGrads.back()[j];
        }
        
        FinalLayer_.setWeights(FinalWeights);
        FinalLayer_.setBiases(FinalBiases);
        
        if (Verbose && (epoch % 20 == 0 || epoch == Epochs - 1)) {
            std::cout << "epoch " << epoch << "/" << Epochs << ", loss: " << CurrentLoss << std::endl;
        }
    }
    
    IsTrained_ = true;
}

// use the trained gnn to predict node features or classifications
std::vector<std::vector<float>> GNN::predict(const std::vector<std::vector<float>>& NodeFeatures,
                                            const std::vector<std::vector<float>>& AdjacencyMatrix) {
    if (!IsTrained_) {
        std::cout << "model not trained yet." << std::endl;
        return {};
    }
    
    auto [Predictions, _] = forward(NodeFeatures, AdjacencyMatrix);
    return Predictions;
}

} // namespace CPP_ML

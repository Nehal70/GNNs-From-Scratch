#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <random>
#include <cmath>
#include <algorithm>

namespace CPP_ML {

class GraphConvolutionLayer {
public:
    // constructor that initializes weights and bias for this layer
    // similar to python's GraphConvolutionLayer.__init__
    GraphConvolutionLayer(int InputDim, int OutputDim);
    
    // forward pass through graph convolution layer
    // similar to python's GraphConvolutionLayer.forward
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& NodeFeatures,
                                           const std::vector<std::vector<float>>& AdjacencyMatrix);
    
    // getter methods for weights and biases
    const std::vector<std::vector<float>>& getWeights() const { return Weights_; }
    const std::vector<float>& getBiases() const { return Biases_; }
    
    // setter methods for weights and biases (used in backpropagation)
    void setWeights(const std::vector<std::vector<float>>& Weights) { Weights_ = Weights; }
    void setBiases(const std::vector<float>& Biases) { Biases_ = Biases; }

private:
    std::vector<std::vector<float>> Weights_;  // weight matrix
    std::vector<float> Biases_;                // bias vector
};

class GNN {
public:
    // constructor that sets up the graph neural network with the specified architecture and hyperparameters
    // similar to python version: GNN(input_dim, hidden_dims, output_dim, task_type, learning_rate)
    GNN(int InputDim, const std::vector<int>& HiddenDims, int OutputDim, 
        const std::string& TaskType = "classification", float LearningRate = 0.01f);
    
    // train the gnn on graph data using sophisticated graph-aware backpropagation
    // similar to python version: gnn.train(node_features, adjacency_matrix, labels, epochs, verbose)
    void train(const std::vector<std::vector<float>>& NodeFeatures,
              const std::vector<std::vector<float>>& AdjacencyMatrix,
              const std::vector<std::vector<float>>& Labels, int Epochs = 100, bool Verbose = true);
    
    // use the trained gnn to predict node features or classifications
    // similar to python version: gnn.predict(node_features, adjacency_matrix)
    std::vector<std::vector<float>> predict(const std::vector<std::vector<float>>& NodeFeatures,
                                           const std::vector<std::vector<float>>& AdjacencyMatrix);
    
    // forward pass through the gnn (made public for testing and inspection)
    // similar to python version: gnn.forward(node_features, adjacency_matrix)
    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>> 
    forward(const std::vector<std::vector<float>>& NodeFeatures,
            const std::vector<std::vector<float>>& AdjacencyMatrix);
    
    // calculate loss based on task type
    // similar to python version: gnn._calculate_loss(y_true, y_pred)
    float calculateLoss(const std::vector<std::vector<float>>& Predictions,
                       const std::vector<std::vector<float>>& Labels);
    
    // functions to inspect the gnns current configuration and training state
    // similar to python version properties
    const std::vector<int>& getLayerSizes() const { return LayerSizes_; }
    const std::string& getTaskType() const { return TaskType_; }
    float getLearningRate() const { return LearningRate_; }
    bool isTrained() const { return IsTrained_; }

private:
    // architecture parameters - similar to python version
    int InputDim_;
    std::vector<int> HiddenDims_;
    int OutputDim_;
    std::string TaskType_;
    float LearningRate_;
    bool IsTrained_;
    
    // layer sizes for internal use
    std::vector<int> LayerSizes_;
    
    // graph convolution layers
    std::vector<GraphConvolutionLayer> GNNLayers_;
    GraphConvolutionLayer FinalLayer_;
    
    // internal functions that implement the core graph neural network operations
    // similar to python version methods
    void initializeWeights();
    std::vector<std::vector<float>> applyActivation(const std::vector<std::vector<float>>& Input, 
                                                   const std::string& Activation);
    
    // sophisticated backpropagation methods
    std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<float>>> 
    computeGraphGradients(const std::vector<std::vector<float>>& NodeFeatures,
                         const std::vector<std::vector<float>>& AdjacencyMatrix,
                         const std::vector<std::vector<float>>& Labels,
                         const std::vector<std::vector<std::vector<float>>>& Activations);
    
    std::vector<std::vector<float>> backpropGraphConv(const std::vector<std::vector<float>>& GradOutput,
                                                     const std::vector<std::vector<float>>& AdjacencyMatrix,
                                                     const std::vector<std::vector<float>>& Weights);
    
    std::vector<std::vector<float>> reluDerivative(const std::vector<std::vector<float>>& Input);
};

} // namespace CPP_ML

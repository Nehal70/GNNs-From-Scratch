#pragma once

#include <vector>
#include <string>
#include <stdexcept>

namespace CUDA_ML {

class NeuralNetwork {
public:
    // constructor that sets up the neural network architecture and training parameters
    NeuralNetwork(const std::vector<int>& layers, 
                 const std::vector<std::string>& activations,
                 float learning_rate = 0.01f);
    
    // cleanup gpu memory when the neural network is destroyed
    ~NeuralNetwork();
    
    // train the neural network on provided data using gradient descent over multiple epochs
    void train(const std::vector<std::vector<float>>& X_train,
              const std::vector<std::vector<float>>& y_train,
              int epochs = 100, bool verbose = true);
    
    // use the trained network to make predictions on new data
    std::vector<std::vector<float>> predict(const std::vector<std::vector<float>>& X_test);
    
    // functions to inspect the networks current state and parameters
    const std::vector<int>& getLayers() const { return layers_; }
    const std::vector<std::string>& getActivations() const { return activations_; }
    float getLearningRate() const { return learning_rate_; }
    bool isTrained() const { return is_trained_; }

private:
    std::vector<int> layers_;
    std::vector<std::string> activations_;
    float learning_rate_;
    bool is_trained_;
    
    // pointers to gpu memory where we store the learned weights and biases for each layer
    std::vector<float*> d_weights_;
    std::vector<float*> d_biases_;
    
    // internal functions that handle the core neural network operations without exposing implementation details
    void initializeWeights();
    void forward(const float* d_input, int batch_size, 
                std::vector<float*>& d_activations,
                std::vector<float*>& d_z_values);
    void backward(const float* d_input, const float* d_y_true,
                 const std::vector<float*>& d_activations,
                 const std::vector<float*>& d_z_values,
                 int batch_size);
    float calculateLoss(const float* d_y_true, const float* d_y_pred, int batch_size);
    
    // apply activation functions and their derivatives during forward and backward passes
    void applyActivation(const float* input, float* output, const std::string& activation, int size);
    void applyActivationDerivative(const float* activated_values, float* derivatives, const std::string& activation, int size);
};

} // namespace CUDA_ML


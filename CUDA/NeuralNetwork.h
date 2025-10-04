#pragma once

#include <vector>
#include <string>
#include <stdexcept>

namespace CUDA_ML {

class NeuralNetwork {
public:
    // Constructor
    NeuralNetwork(const std::vector<int>& layers, 
                 const std::vector<std::string>& activations,
                 float learning_rate = 0.01f);
    
    // Destructor
    ~NeuralNetwork();
    
    // Training function
    void train(const std::vector<std::vector<float>>& X_train,
              const std::vector<std::vector<float>>& y_train,
              int epochs = 100, bool verbose = true);
    
    // Prediction function
    std::vector<std::vector<float>> predict(const std::vector<std::vector<float>>& X_test);
    
    // Getters
    const std::vector<int>& getLayers() const { return layers_; }
    const std::vector<std::string>& getActivations() const { return activations_; }
    float getLearningRate() const { return learning_rate_; }
    bool isTrained() const { return is_trained_; }

private:
    std::vector<int> layers_;
    std::vector<std::string> activations_;
    float learning_rate_;
    bool is_trained_;
    
    // CUDA device memory pointers for weights and biases
    std::vector<float*> d_weights_;
    std::vector<float*> d_biases_;
    
    // Helper functions
    void initializeWeights();
    void forward(const float* d_input, int batch_size, 
                std::vector<float*>& d_activations,
                std::vector<float*>& d_z_values);
    void backward(const float* d_input, const float* d_y_true,
                 const std::vector<float*>& d_activations,
                 const std::vector<float*>& d_z_values,
                 int batch_size);
    float calculateLoss(const float* d_y_true, const float* d_y_pred, int batch_size);
    
    // activation function helpers
    void applyActivation(const float* input, float* output, const std::string& activation, int size);
    void applyActivationDerivative(const float* activated_values, float* derivatives, const std::string& activation, int size);
};

} // namespace CUDA_ML


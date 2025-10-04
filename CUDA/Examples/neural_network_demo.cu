#include "NeuralNetwork.h"
#include "Common/Utils.h"
#include <iostream>
#include <vector>
#include <chrono>

using namespace CUDA_ML;

int main() {
    std::cout << "=" << std::string(60, '=') << std::endl;
    std::cout << "CUDA NEURAL NETWORK DEMO" << std::endl;
    std::cout << "=" << std::string(60, '=') << std::endl;
    
    // Print CUDA device info
    Utils::printDeviceInfo();
    
    // Generate synthetic classification data
    std::cout << "\n📊 STEP 1: Generating Synthetic Classification Data" << std::endl;
    std::cout << "-" << std::string(40, '-') << std::endl;
    
    int n_train = 500;
    int n_test = 100;
    int input_size = 4;
    int num_classes = 3;
    
    // Generate random features
    std::vector<std::vector<float>> X_train = Utils::generateRandomMatrix(n_train, input_size, -2.0f, 2.0f);
    std::vector<std::vector<float>> X_test = Utils::generateRandomMatrix(n_test, input_size, -2.0f, 2.0f);
    
    // Generate labels based on simple rules
    std::vector<std::vector<float>> y_train(n_train, std::vector<float>(num_classes, 0.0f));
    std::vector<std::vector<float>> y_test(n_test, std::vector<float>(num_classes, 0.0f));
    
    for (int i = 0; i < n_train; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < input_size; ++j) {
            sum += X_train[i][j];
        }
        int class_idx = static_cast<int>(std::abs(sum)) % num_classes;
        y_train[i][class_idx] = 1.0f;
    }
    
    for (int i = 0; i < n_test; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < input_size; ++j) {
            sum += X_test[i][j];
        }
        int class_idx = static_cast<int>(std::abs(sum)) % num_classes;
        y_test[i][class_idx] = 1.0f;
    }
    
    std::cout << "✅ Training samples: " << n_train << std::endl;
    std::cout << "✅ Test samples: " << n_test << std::endl;
    std::cout << "✅ Input features: " << input_size << std::endl;
    std::cout << "✅ Output classes: " << num_classes << std::endl;
    
    // Initialize CUDA Neural Network
    std::cout << "\n🔧 STEP 2: Initializing CUDA Neural Network" << std::endl;
    std::cout << "-" << std::string(40, '-') << std::endl;
    
    std::vector<int> layers = {input_size, 8, 6, num_classes};
    std::vector<std::string> activations = {"relu", "relu", "softmax"};
    float learning_rate = 0.1f;
    
    NeuralNetwork model(layers, activations, learning_rate);
    
    std::cout << "✅ Network architecture: [";
    for (size_t i = 0; i < layers.size(); ++i) {
        std::cout << layers[i];
        if (i < layers.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "✅ Activations: ";
    for (size_t i = 0; i < activations.size(); ++i) {
        std::cout << activations[i];
        if (i < activations.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
    std::cout << "✅ Learning rate: " << learning_rate << std::endl;
    
    // Train the model
    std::cout << "\n🎯 STEP 3: Training CUDA Neural Network" << std::endl;
    std::cout << "-" << std::string(40, '-') << std::endl;
    
    int epochs = 50;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    model.train(X_train, y_train, epochs, true);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "✅ Training completed!" << std::endl;
    std::cout << "✅ Training time: " << duration.count() << " ms" << std::endl;
    
    // Make predictions
    std::cout << "\n🔮 STEP 4: Making Predictions" << std::endl;
    std::cout << "-" << std::string(40, '-') << std::endl;
    
    auto pred_start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> predictions = model.predict(X_test);
    auto pred_end = std::chrono::high_resolution_clock::now();
    auto pred_duration = std::chrono::duration_cast<std::chrono::microseconds>(pred_end - pred_start);
    
    std::cout << "✅ Predictions completed!" << std::endl;
    std::cout << "✅ Prediction time: " << pred_duration.count() << " μs" << std::endl;
    
    // Calculate accuracy
    std::cout << "\n📈 STEP 5: Performance Metrics" << std::endl;
    std::cout << "-" << std::string(40, '-') << std::endl;
    
    int correct = 0;
    for (int i = 0; i < n_test; ++i) {
        int predicted_class = std::distance(predictions[i].begin(), 
                                          std::max_element(predictions[i].begin(), predictions[i].end()));
        int true_class = std::distance(y_test[i].begin(), 
                                     std::max_element(y_test[i].begin(), y_test[i].end()));
        if (predicted_class == true_class) {
            correct++;
        }
    }
    
    float accuracy = static_cast<float>(correct) / n_test;
    std::cout << "✅ Test Accuracy: " << accuracy << " (" << (accuracy * 100.0f) << "%)" << std::endl;
    std::cout << "✅ Correct Predictions: " << correct << "/" << n_test << std::endl;
    
    // Show sample predictions
    std::cout << "\n📋 STEP 6: Sample Predictions" << std::endl;
    std::cout << "-" << std::string(40, '-') << std::endl;
    
    std::cout << "Sample predictions (first 5):" << std::endl;
    std::cout << "Sample\tPredicted\tActual\t\tCorrect" << std::endl;
    std::cout << "-" << std::string(40, '-') << std::endl;
    
    for (int i = 0; i < std::min(5, n_test); ++i) {
        int predicted_class = std::distance(predictions[i].begin(), 
                                          std::max_element(predictions[i].begin(), predictions[i].end()));
        int true_class = std::distance(y_test[i].begin(), 
                                     std::max_element(y_test[i].begin(), y_test[i].end()));
        
        std::cout << i + 1 << "\tClass " << predicted_class << "\t\tClass " << true_class << "\t\t";
        std::cout << (predicted_class == true_class ? "✅" : "❌") << std::endl;
    }
    
    // Summary
    std::cout << "\n🎉 STEP 7: Summary" << std::endl;
    std::cout << "-" << std::string(40, '-') << std::endl;
    std::cout << "✅ CUDA Neural Network successfully implemented!" << std::endl;
    std::cout << "✅ Final test accuracy: " << (accuracy * 100.0f) << "%" << std::endl;
    std::cout << "✅ Training time: " << duration.count() << " ms" << std::endl;
    std::cout << "✅ Prediction time: " << pred_duration.count() << " μs" << std::endl;
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "CUDA NEURAL NETWORK DEMO COMPLETED!" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    return 0;
}




/**
 * Example: Using gnns_ml as a C++ Library
 * 
 * This demonstrates how to use the gnns_ml library in your own C++ projects.
 * 
 * To build:
 *   mkdir build && cd build
 *   cmake ..
 *   make
 *   ./cpp_library_example
 * 
 * Or if linking manually:
 *   g++ -I../C++ -I../CUDA -L../build/lib -lgnns_ml_cpp cpp_library_example.cpp -o example
 */

#include <iostream>
#include <vector>
#include <iomanip>

// Include the library headers
#include "GNN.h"  // From C++ folder
#include "NeuralNetwork.h"
#include "LinearRegression.h"
#include "GraphAlgorithms.h"

using namespace CPP_ML;

// Helper function to print a 2D vector
template<typename T>
void printMatrix(const std::vector<std::vector<T>>& matrix, const std::string& name) {
    std::cout << "\n" << name << ":\n";
    for (size_t i = 0; i < matrix.size() && i < 5; ++i) {  // Print first 5 rows
        std::cout << "  Row " << i << ": ";
        for (size_t j = 0; j < matrix[i].size() && j < 5; ++j) {  // Print first 5 cols
            std::cout << std::fixed << std::setprecision(3) << matrix[i][j] << " ";
        }
        std::cout << "...\n";
    }
}

// Example 1: Using GNN
void exampleGNN() {
    std::cout << "============================================================" << std::endl;
    std::cout << "Example 1: Graph Neural Network for Node Classification" << std::endl;
    std::cout << "============================================================" << std::endl;
    
    // Create sample graph data
    int num_nodes = 30;
    int feature_dim = 4;
    
    std::vector<std::vector<float>> node_features(num_nodes);
    std::vector<std::vector<float>> adjacency(num_nodes, std::vector<float>(num_nodes, 0.0f));
    
    // Initialize random features
    for (int i = 0; i < num_nodes; ++i) {
        node_features[i].resize(feature_dim);
        for (int j = 0; j < feature_dim; ++j) {
            node_features[i][j] = (float)(rand() % 200 - 100) / 100.0f;
        }
    }
    
    // Create random adjacency matrix
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = i + 1; j < num_nodes; ++j) {
            if ((rand() % 100) < 30) {  // 30% connection probability
                adjacency[i][j] = 1.0f;
                adjacency[j][i] = 1.0f;
            }
        }
    }
    
    // Create random labels (one-hot encoded)
    std::vector<std::vector<float>> labels(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        int class_label = rand() % 3;
        labels[i].resize(3, 0.0f);
        labels[i][class_label] = 1.0f;
    }
    
    std::cout << "\nCreating GNN with:" << std::endl;
    std::cout << "  Input dim: " << feature_dim << std::endl;
    std::cout << "  Hidden dims: [32, 16]" << std::endl;
    std::cout << "  Output dim: 3 (classification)" << std::endl;
    
    // Create and train GNN
    GNN gnn(feature_dim, {32, 16}, 3, "classification", 0.01f);
    
    std::cout << "\nTraining GNN...";
    gnn.train(node_features, adjacency, labels, 50, false);  // Don't print verbose
    std::cout << " completed!" << std::endl;
    
    // Make predictions
    auto predictions = gnn.predict(node_features, adjacency);
    std::cout << "Predictions made for " << predictions.size() << " nodes" << std::endl;
    
    // Calculate accuracy
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        int pred_class = 0;
        int true_class = 0;
        float max_pred = predictions[i][0];
        float max_true = labels[i][0];
        
        for (size_t j = 1; j < 3; ++j) {
            if (predictions[i][j] > max_pred) {
                max_pred = predictions[i][j];
                pred_class = j;
            }
            if (labels[i][j] > max_true) {
                max_true = labels[i][j];
                true_class = j;
            }
        }
        
        if (pred_class == true_class) correct++;
    }
    
    float accuracy = (float)correct / predictions.size();
    std::cout << "Accuracy: " << std::fixed << std::setprecision(2) 
              << accuracy * 100 << "%" << std::endl;
}

// Example 2: Using Linear Regression
void exampleLinearRegression() {
    std::cout << "\n============================================================" << std::endl;
    std::cout << "Example 2: Linear Regression" << std::endl;
    std::cout << "============================================================" << std::endl;
    
    // Create simple synthetic data: y = 2x + 1 + noise
    int n_samples = 100;
    std::vector<std::vector<float>> X(n_samples);
    std::vector<float> y(n_samples);
    
    for (int i = 0; i < n_samples; ++i) {
        float x_val = (float)i / 10.0f - 5.0f;
        X[i] = {x_val};
        y[i] = 2.0f * x_val + 1.0f + (float)(rand() % 20 - 10) / 50.0f;
    }
    
    std::cout << "\nTraining linear regression on " << n_samples << " samples..." << std::endl;
    
    LinearRegression lr(0.01f);
    lr.train(X, y, 500, false);
    
    auto predictions = lr.predict(X);
    
    // Calculate MSE
    float mse = 0.0f;
    for (size_t i = 0; i < predictions.size(); ++i) {
        float error = predictions[i] - y[i];
        mse += error * error;
    }
    mse /= predictions.size();
    
    std::cout << "Mean Squared Error: " << std::fixed << std::setprecision(6) << mse << std::endl;
    std::cout << "Learned weight: " << std::fixed << std::setprecision(2) 
              << lr.getWeights()[0][0] << " (expected ~2.0)" << std::endl;
    std::cout << "Learned bias: " << std::fixed << std::setprecision(2) 
              << lr.getBias() << " (expected ~1.0)" << std::endl;
}

int main() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "GNNs-From-Scratch C++ Library Usage Examples" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    std::cout << "\nThis demonstrates using gnns_ml as a C++ library." << std::endl;
    std::cout << "All implementations are done from scratch.\n" << std::endl;
    
    try {
        exampleLinearRegression();
        exampleGNN();
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "All examples completed successfully!" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}


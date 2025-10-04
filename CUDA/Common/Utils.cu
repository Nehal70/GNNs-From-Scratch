#include "Utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>

namespace CUDA_ML {

void Utils::checkCudaError(cudaError_t error, const std::string& message) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " - " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("CUDA operation failed");
    }
}

void Utils::printDeviceInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    std::cout << "CUDA Device Count: " << deviceCount << std::endl;
    
    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        
        std::cout << "\nDevice " << device << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Shared Memory per Block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Thread Dimensions: [" << deviceProp.maxThreadsDim[0] 
                  << ", " << deviceProp.maxThreadsDim[1] 
                  << ", " << deviceProp.maxThreadsDim[2] << "]" << std::endl;
        std::cout << "  Max Grid Dimensions: [" << deviceProp.maxGridSize[0] 
                  << ", " << deviceProp.maxGridSize[1] 
                  << ", " << deviceProp.maxGridSize[2] << "]" << std::endl;
    }
}

void Utils::setRandomSeed(unsigned int seed) {
    srand(seed);
    // Note: For CUDA, you might also want to set curand seed if using cuRAND
}

std::vector<float> Utils::generateRandomData(int size, float min_val, float max_val) {
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min_val, max_val);
    
    for (auto& value : data) {
        value = dist(gen);
    }
    
    return data;
}

std::vector<std::vector<float>> Utils::generateRandomMatrix(int rows, int cols, float min_val, float max_val) {
    std::vector<std::vector<float>> matrix(rows);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min_val, max_val);
    
    for (int i = 0; i < rows; ++i) {
        matrix[i].resize(cols);
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = dist(gen);
        }
    }
    
    return matrix;
}

float Utils::calculateAccuracy(const std::vector<float>& predictions, const std::vector<float>& labels) {
    if (predictions.size() != labels.size()) {
        throw std::invalid_argument("Predictions and labels must have same size");
    }
    
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (std::abs(predictions[i] - labels[i]) < 0.5f) {  // Simple threshold for classification
            correct++;
        }
    }
    
    return static_cast<float>(correct) / predictions.size();
}

void Utils::printMatrix(const std::vector<std::vector<float>>& matrix, const std::string& name) {
    std::cout << name << ":" << std::endl;
    for (const auto& row : matrix) {
        for (const auto& value : row) {
            std::cout << std::fixed << std::setprecision(4) << value << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void Utils::printVector(const std::vector<float>& vec, const std::string& name) {
    std::cout << name << ": [";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << vec[i];
        if (i < vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

} // namespace CUDA_ML




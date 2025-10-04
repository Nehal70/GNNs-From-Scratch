#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <iomanip>

namespace CUDA_ML {

class Utils {
public:
    // CUDA error checking
    static void checkCudaError(cudaError_t error, const std::string& message);
    
    // Device information
    static void printDeviceInfo();
    
    // Random number generation
    static void setRandomSeed(unsigned int seed);
    static std::vector<float> generateRandomData(int size, float min_val = -1.0f, float max_val = 1.0f);
    static std::vector<std::vector<float>> generateRandomMatrix(int rows, int cols, 
                                                               float min_val = -1.0f, float max_val = 1.0f);
    
    // Evaluation metrics
    static float calculateAccuracy(const std::vector<float>& predictions, const std::vector<float>& labels);
    
    // Debugging and visualization
    static void printMatrix(const std::vector<std::vector<float>>& matrix, const std::string& name = "Matrix");
    static void printVector(const std::vector<float>& vec, const std::string& name = "Vector");
};

} // namespace CUDA_ML




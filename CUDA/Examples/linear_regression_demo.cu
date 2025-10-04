// linear_regression_demo.cu
#include "LinearRegression.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

int main() {
    std::cout << "=" << std::string(60, '=') << std::endl;
    std::cout << "CUDA LINEAR REGRESSION DEMO" << std::endl;
    std::cout << "=" << std::string(60, '=') << std::endl;
    
    // Generate synthetic data: y = 2x + 3 + noise
    std::cout << "\n📊 STEP 1: Generating Synthetic Data" << std::endl;
    std::cout << "-" << std::string(40, '-') << std::endl;
    
    int n_train = 1000;
    int n_test = 200;
    
    std::vector<float> x_train(n_train);
    std::vector<float> y_train(n_train);
    std::vector<float> x_test(n_test);
    std::vector<float> y_test(n_test);
    
    // Generate random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> x_dist(-5.0f, 5.0f);
    std::normal_distribution<float> noise(0.0f, 0.5f);
    
    // Training data
    for (int i = 0; i < n_train; ++i) {
        x_train[i] = x_dist(gen);
        y_train[i] = 2.0f * x_train[i] + 3.0f + noise(gen);
    }
    
    // Test data
    for (int i = 0; i < n_test; ++i) {
        x_test[i] = x_dist(gen);
        y_test[i] = 2.0f * x_test[i] + 3.0f + noise(gen);
    }
    
    std::cout << "✅ Training samples: " << n_train << std::endl;
    std::cout << "✅ Test samples: " << n_test << std::endl;
    std::cout << "✅ True relationship: y = 2x + 3 + noise" << std::endl;
    
    // Initialize CUDA Linear Regression model
    std::cout << "\n🔧 STEP 2: Initializing CUDA Linear Regression" << std::endl;
    std::cout << "-" << std::string(40, '-') << std::endl;
    
    float learning_rate = 0.01f;
    LinearRegression model(learning_rate);
    std::cout << "✅ Learning rate: " << learning_rate << std::endl;
    
    // Train the model
    std::cout << "\n🎯 STEP 3: Training CUDA Linear Regression" << std::endl;
    std::cout << "-" << std::string(40, '-') << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    model.train(x_train, y_train);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "✅ Training completed!" << std::endl;
    std::cout << "✅ Training time: " << duration.count() << " ms" << std::endl;
    std::cout << "✅ Learned slope: " << model.getSlope() << " (true: 2.0)" << std::endl;
    std::cout << "✅ Learned intercept: " << model.getIntercept() << " (true: 3.0)" << std::endl;
    
    // Make predictions
    std::cout << "\n🔮 STEP 4: Making Predictions" << std::endl;
    std::cout << "-" << std::string(40, '-') << std::endl;
    
    auto pred_start = std::chrono::high_resolution_clock::now();
    std::vector<float> predictions = model.predict(x_test);
    auto pred_end = std::chrono::high_resolution_clock::now();
    auto pred_duration = std::chrono::duration_cast<std::chrono::microseconds>(pred_end - pred_start);
    
    std::cout << "✅ Predictions completed!" << std::endl;
    std::cout << "✅ Prediction time: " << pred_duration.count() << " μs" << std::endl;
    
    // Calculate MSE
    std::cout << "\n📈 STEP 5: Performance Metrics" << std::endl;
    std::cout << "-" << std::string(40, '-') << std::endl;
    
    float mse = 0.0f;
    for (int i = 0; i < n_test; ++i) {
        float error = y_test[i] - predictions[i];
        mse += error * error;
    }
    mse /= n_test;
    
    std::cout << "✅ Mean Squared Error: " << mse << std::endl;
    
    // Show sample predictions
    std::cout << "\n📋 STEP 6: Sample Predictions" << std::endl;
    std::cout << "-" << std::string(40, '-') << std::endl;
    
    std::cout << "Sample predictions (first 10):" << std::endl;
    std::cout << "X\t\tTrue Y\t\tPredicted Y\t\tError" << std::endl;
    std::cout << "-" << std::string(50, '-') << std::endl;
    
    for (int i = 0; i < std::min(10, n_test); ++i) {
        float error = std::abs(y_test[i] - predictions[i]);
        std::cout << std::fixed << std::setprecision(3) 
                  << x_test[i] << "\t\t" 
                  << y_test[i] << "\t\t" 
                  << predictions[i] << "\t\t" 
                  << error << std::endl;
    }
    
    // Summary
    std::cout << "\n🎉 STEP 7: Summary" << std::endl;
    std::cout << "-" << std::string(40, '-') << std::endl;
    std::cout << "✅ CUDA Linear Regression successfully implemented!" << std::endl;
    std::cout << "✅ Model learned: y = " << model.getSlope() << "x + " << model.getIntercept() << std::endl;
    std::cout << "✅ Final MSE: " << mse << std::endl;
    std::cout << "✅ Training time: " << duration.count() << " ms" << std::endl;
    std::cout << "✅ Prediction time: " << pred_duration.count() << " μs" << std::endl;
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "CUDA LINEAR REGRESSION DEMO COMPLETED!" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    return 0;
}
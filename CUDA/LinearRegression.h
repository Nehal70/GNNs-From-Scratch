// LinearRegression.h
#pragma once
#include <vector>

class LinearRegression {
public:
    // Constructor - user can set learning rate
    LinearRegression(float learning_rate = 0.01f);
    
    // Training - user provides data
    void train(const std::vector<float>& x_train, const std::vector<float>& y_train);
    
    // Prediction - user gets results
    std::vector<float> predict(const std::vector<float>& x_test);
    
    // Getters - user can check results
    float getSlope() const { return slope_; }
    float getIntercept() const { return intercept_; }
    bool isTrained() const { return is_trained_; }

private:
    // Internal state - only class can modify
    float learning_rate_;
    float slope_;
    float intercept_;
    bool is_trained_;
};




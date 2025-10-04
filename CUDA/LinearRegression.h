// LinearRegression.h
#pragma once
#include <vector>

class LinearRegression {
public:
    // constructor that sets up the learning rate for gradient descent optimization
    LinearRegression(float learning_rate = 0.01f);
    
    // train the linear regression model using cuda-accelerated gradient descent
    void train(const std::vector<float>& x_train, const std::vector<float>& y_train);
    
    // make predictions using the learned slope and intercept parameters
    std::vector<float> predict(const std::vector<float>& x_test);
    
    // functions to inspect the learned parameters after training is complete
    float getSlope() const { return slope_; }
    float getIntercept() const { return intercept_; }
    bool isTrained() const { return is_trained_; }

private:
    // internal parameters that get learned during training and manage the models state
    float learning_rate_;
    float slope_;
    float intercept_;
    bool is_trained_;
};




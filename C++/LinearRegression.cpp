#include "LinearRegression.h"
#include <numeric>
#include <cmath>

// constructor
LinearRegression::LinearRegression() 
    : slope(0.0), intercept(0.0), is_trained(false) {
}

//destructor, a c++ practise I dont use in python
LinearRegression::~LinearRegression() {
}

// train function matching my python implementation
void LinearRegression::train(const std::vector<double>& x_train, const std::vector<double>& y_train) {
    double x_sum = std::accumulate(x_train.begin(), x_train.end(), 0.0);
    double y_sum = std::accumulate(y_train.begin(), y_train.end(), 0.0);
    
    double xy_sum = 0.0;
    double x2_sum = 0.0;
    for (size_t i = 0; i < x_train.size(); i++) {
        xy_sum += x_train[i] * y_train[i];
        x2_sum += x_train[i] * x_train[i];
    }
    
    int length = x_train.size();
    
    // slope using least squares formula
    slope = (length * xy_sum - x_sum * y_sum) / (length * x2_sum - x_sum * x_sum);
    
    // intercept using least squares formula
    intercept = (y_sum - slope * x_sum) / length;
    
    // update object attributes
    is_trained = true;
}

// predict
std::vector<double> LinearRegression::predict(const std::vector<double>& x) {
    std::vector<double> predictions;
    for (double xi : x) {
        predictions.push_back(slope * xi + intercept);
    }
    return predictions;
}

// mse
double LinearRegression::calculate_mse(const std::vector<double>& y_actual, const std::vector<double>& y_pred) {
    double sum_squared_error = 0.0;
    for (size_t i = 0; i < y_actual.size(); i++) {
        double error = y_actual[i] - y_pred[i];
        sum_squared_error += error * error;
    }
    return sum_squared_error / y_actual.size();
}

// r-squared
double LinearRegression::calculate_r_squared(const std::vector<double>& y_actual, const std::vector<double>& y_pred) {
    double y_mean = std::accumulate(y_actual.begin(), y_actual.end(), 0.0) / y_actual.size();
    
    double ss_res = 0.0;  
    double ss_tot = 0.0; 
    
    for (size_t i = 0; i < y_actual.size(); i++) {
        ss_res += (y_actual[i] - y_pred[i]) * (y_actual[i] - y_pred[i]);
        ss_tot += (y_actual[i] - y_mean) * (y_actual[i] - y_mean);
    }
    
    return 1.0 - (ss_res / ss_tot);
}

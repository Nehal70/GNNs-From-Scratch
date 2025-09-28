#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <vector>
#include <string>

class LinearRegression {
private:
    double slope;
    double intercept;
    bool is_trained;

public:
    LinearRegression();  // Constructor
    ~LinearRegression(); // Destructor
    
    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>> 
    load_data(const std::string& filename = "linear_regression_data.csv");
    void train(const std::vector<double>& x_train, const std::vector<double>& y_train);
    std::vector<double> predict(const std::vector<double>& x);
    double calculate_mse(const std::vector<double>& y_actual, const std::vector<double>& y_pred);
    double calculate_r_squared(const std::vector<double>& y_actual, const std::vector<double>& y_pred);

};

#endif

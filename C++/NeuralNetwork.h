//C++ Implementation that uses my python Neural Network logic
#pragma once

#include <vector>
#include <string>
#include <utility>

// activations: relu, sigmoid, tanh, leaky_relu, softmax

class NeuralNetwork {
public:
    //constructor
    NeuralNetwork(const std::vector<int>& layers,
                  const std::vector<std::string>& activations,
                  double learningRate = 0.01);

    //load from csv 
    static std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<std::vector<double>>>
    loadData(const std::string& trainFilename = "train_data_nn.csv",
             const std::string& testFilename = "test_data_nn.csv",
             int inputSize = -1,
             int outputSize = -1);

    // forward pass
    std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<std::vector<double>>>>
    forward(const std::vector<std::vector<double>>& X) const;

    // Train with full-batch gradient descent for a fixed number of epochs
    std::vector<double> train(const std::vector<std::vector<double>>& X_train,
                              const std::vector<std::vector<double>>& y_train,
                              int epochs = 100,
                              bool verbose = true);

    // Prediction function (post training)
    std::vector<std::vector<double>> predict(const std::vector<std::vector<double>>& X) const;

    // Public metadata
    const std::vector<int>& getLayers() const { return layers; }
    const std::vector<std::string>& getActivations() const { return activations; }
    double getLearningRate() const { return learningRate; }
    bool trained() const { return isTrained; }

private:
    // Parameters
    std::vector<int> layers;
    std::vector<std::string> activations;
    double learningRate;
    std::vector<std::vector<std::vector<double>>> weights; 
    std::vector<std::vector<double>> biases;               
    bool isTrained;

    
    static double randomNormalHe(int fanIn);
    static double sigmoid(double z);
    static double tanhFn(double z);
    static double relu(double z);
    static double leakyRelu(double z);

    static double sigmoidDerivFromActivated(double a);
    static double tanhDerivFromActivated(double a);
    static double reluDerivFromActivated(double a);
    static double leakyReluDerivFromActivated(double a);

    static std::vector<std::vector<double>> softmax(const std::vector<std::vector<double>>& Z);

    static std::vector<std::vector<double>> matmul(const std::vector<std::vector<double>>& A,
                                                   const std::vector<std::vector<double>>& B);

    static std::vector<std::vector<double>> matvecAdd(const std::vector<std::vector<double>>& A,
                                                      const std::vector<double>& b);

    static std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& A);

    static std::vector<std::vector<double>> applyActivation(const std::vector<std::vector<double>>& Z,
                                                            const std::string& activation);

    static std::vector<std::vector<double>> activationDerivativeFromActivated(const std::vector<std::vector<double>>& A,
                                                                             const std::string& activation);

    static std::vector<std::vector<double>> hadamard(const std::vector<std::vector<double>>& A,
                                                     const std::vector<std::vector<double>>& B);

    static std::vector<std::vector<double>> scalarMat(const std::vector<std::vector<double>>& A, double s);

    static std::vector<std::vector<double>> add(const std::vector<std::vector<double>>& A,
                                                const std::vector<std::vector<double>>& B);

    static std::vector<std::vector<double>> subtract(const std::vector<std::vector<double>>& A,
                                                     const std::vector<std::vector<double>>& B);

    // Loss function
    double calculateLoss(const std::vector<std::vector<double>>& yTrue,
                         const std::vector<std::vector<double>>& yPred) const;

    // Backpropagation function
    void backward(const std::vector<std::vector<double>>& X,
                  const std::vector<std::vector<double>>& y,
                  const std::vector<std::vector<std::vector<double>>>& activationsAll,
                  const std::vector<std::vector<std::vector<double>>>& zValuesAll);
};


int main();



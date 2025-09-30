#include "NeuralNetwork.h"

#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <limits>

namespace {
    std::mt19937& globalRng() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        return gen;
    }
}

// constructor
NeuralNetwork::NeuralNetwork(const std::vector<int>& layers,
                             const std::vector<std::string>& activations,
                             double learningRate)
    : layers(layers), activations(activations), learningRate(learningRate), isTrained(false) {
    if (layers.size() < 2) {
        throw std::invalid_argument("At least input and output layers are required");
    }
    if (activations.size() != layers.size() - 1) {
        throw std::invalid_argument("activations size must equal layers.size()-1");
    }
    

    //initialising weights and biases
    weights.resize(layers.size() - 1);
    biases.resize(layers.size() - 1);
    for (size_t i = 0; i + 1 < layers.size(); ++i) {
        int inSize = layers[i];
        int outSize = layers[i + 1];
        weights[i].assign(inSize, std::vector<double>(outSize, 0.0));
        biases[i].assign(outSize, 0.0);
        double stddev = std::sqrt(2.0 / static_cast<double>(inSize));
        std::normal_distribution<double> dist(0.0, stddev);
        auto& gen = globalRng();
        for (int r = 0; r < inSize; ++r) {
            for (int c = 0; c < outSize; ++c) {
                weights[i][r][c] = dist(gen);
            }
        }
    }
}

//activation functions
double NeuralNetwork::sigmoid(double z) {
     return 1.0 / (1.0 + std::exp(-z)); 
    }

double NeuralNetwork::tanhFn(double z) {
     return std::tanh(z); 
    }

double NeuralNetwork::relu(double z) {
     return z > 0.0 ? z : 0.0; 
    }

double NeuralNetwork::leakyRelu(double z) { 
    return z > 0.0 ? z : 0.01 * z; 
}

//activation derivatives
double NeuralNetwork::sigmoidDerivFromActivated(double a) {
     return a * (1.0 - a); 
    }

double NeuralNetwork::tanhDerivFromActivated(double a) {
     return 1.0 - a * a; 
    }

double NeuralNetwork::reluDerivFromActivated(double a) {
     return a > 0.0 ? 1.0 : 0.0; 
    }

double NeuralNetwork::leakyReluDerivFromActivated(double a) {
     return a > 0.0 ? 1.0 : 0.01; 
    }


// matrix transpose
std::vector<std::vector<double>> NeuralNetwork::transpose(const std::vector<std::vector<double>>& A) {
    if (A.empty()) return {};
    size_t n = A.size();
    size_t m = A[0].size();
    std::vector<std::vector<double>> T(m, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            T[j][i] = A[i][j];
        }
    }
    return T;
}

// Matrix multiplication
std::vector<std::vector<double>> NeuralNetwork::matmul(const std::vector<std::vector<double>>& A,
                                                       const std::vector<std::vector<double>>& B) {
    size_t n = A.size();
    size_t k = A.empty() ? 0 : A[0].size();
    size_t m = B.empty() ? 0 : B[0].size();
    if (B.size() != k) {
        throw std::invalid_argument("matmul dimension mismatch");
    }
    std::vector<std::vector<double>> C(n, std::vector<double>(m, 0.0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t t = 0; t < k; ++t) {
            double a = A[i][t];
            if (a == 0.0) continue;
            for (size_t j = 0; j < m; ++j) {
                C[i][j] += a * B[t][j];
            }
        }
    }
    return C;
}

// bias vector addition
std::vector<std::vector<double>> NeuralNetwork::matvecAdd(const std::vector<std::vector<double>>& A,
                                                          const std::vector<double>& b) {
    size_t n = A.size();
    size_t m = A.empty() ? 0 : A[0].size();
    if (b.size() != m) throw std::invalid_argument("matvecAdd length mismatch");
    std::vector<std::vector<double>> C = A;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            C[i][j] += b[j];
        }
    }
    return C;
}

// element products
std::vector<std::vector<double>> NeuralNetwork::hadamard(const std::vector<std::vector<double>>& A,
                                                         const std::vector<std::vector<double>>& B) {
    if (A.size() != B.size() || (!A.empty() && A[0].size() != B[0].size()))
        throw std::invalid_argument("hadamard dimension mismatch");
    std::vector<std::vector<double>> C = A;
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[i].size(); ++j) C[i][j] *= B[i][j];
    }
    return C;
}

// Scale a matrix by scalar
std::vector<std::vector<double>> NeuralNetwork::scalarMat(const std::vector<std::vector<double>>& A, double s) {
    std::vector<std::vector<double>> C = A;
    for (auto& row : C) for (auto& v : row) v *= s;
    return C;
}

// Matrix addition
std::vector<std::vector<double>> NeuralNetwork::add(const std::vector<std::vector<double>>& A,
                                                    const std::vector<std::vector<double>>& B) {
    if (A.size() != B.size() || (!A.empty() && A[0].size() != B[0].size()))
        throw std::invalid_argument("add dimension mismatch");
    std::vector<std::vector<double>> C = A;
    for (size_t i = 0; i < A.size(); ++i) for (size_t j = 0; j < A[i].size(); ++j) C[i][j] += B[i][j];
    return C;


}

// Matrix subtraction
std::vector<std::vector<double>> NeuralNetwork::subtract(const std::vector<std::vector<double>>& A,
                                                         const std::vector<std::vector<double>>& B) {
    if (A.size() != B.size() || (!A.empty() && A[0].size() != B[0].size()))
        throw std::invalid_argument("subtract dimension mismatch");
    std::vector<std::vector<double>> C = A;
    for (size_t i = 0; i < A.size(); ++i) for (size_t j = 0; j < A[i].size(); ++j) C[i][j] -= B[i][j];
    return C;
}

// Apply chosen activation per layer
std::vector<std::vector<double>> NeuralNetwork::applyActivation(const std::vector<std::vector<double>>& Z,
                                                               const std::string& activation) {
    std::vector<std::vector<double>> A = Z;
    if (activation == "relu") {
        for (auto& row : A) for (auto& v : row) v = relu(v);
    } else if (activation == "sigmoid") {
        for (auto& row : A) for (auto& v : row) {
            double zc = std::max(-500.0, std::min(500.0, v));
            v = sigmoid(zc);
        }
    } else if (activation == "tanh") {
        for (auto& row : A) for (auto& v : row) v = tanhFn(v);
    } else if (activation == "leaky_relu") {
        for (auto& row : A) for (auto& v : row) v = leakyRelu(v);
    } else if (activation == "softmax") {
        return softmax(Z);
    } else {
        throw std::invalid_argument("Unsupported activation: " + activation);
    }
    return A;
}

// Compute activation derivative
std::vector<std::vector<double>> NeuralNetwork::activationDerivativeFromActivated(
    const std::vector<std::vector<double>>& A, const std::string& activation) {
    std::vector<std::vector<double>> D = A;
    if (activation == "relu") {
        for (auto& row : D) for (auto& v : row) v = reluDerivFromActivated(v);
    } else if (activation == "sigmoid") {
        for (auto& row : D) for (auto& v : row) v = sigmoidDerivFromActivated(v);
    } else if (activation == "tanh") {
        for (auto& row : D) for (auto& v : row) v = tanhDerivFromActivated(v);
    } else if (activation == "leaky_relu") {
        for (auto& row : D) for (auto& v : row) v = leakyReluDerivFromActivated(v);
    } else {
        // For linear/softmax treat as ones
        for (auto& row : D) for (auto& v : row) v = 1.0;
    }
    return D;
}

// Stable row-wise softmax
std::vector<std::vector<double>> NeuralNetwork::softmax(const std::vector<std::vector<double>>& Z) {
    std::vector<std::vector<double>> S = Z;
    for (auto& row : S) {
        double maxv = *std::max_element(row.begin(), row.end());
        double sum = 0.0;
        for (double& v : row) {
            v = std::exp(v - maxv);
            sum += v;
        }
        double inv = 1.0 / sum;
        for (double& v : row) v *= inv;
    }
    return S;
}

// Forward pass 
std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<std::vector<double>>>>
NeuralNetwork::forward(const std::vector<std::vector<double>>& X) const {
    std::vector<std::vector<std::vector<double>>> activationsAll;
    std::vector<std::vector<std::vector<double>>> zValuesAll;
    activationsAll.push_back(X); // A0 = X
    std::vector<std::vector<double>> current = X;
    for (size_t layer = 0; layer < weights.size(); ++layer) {
        auto Z = matvecAdd(matmul(current, weights[layer]), biases[layer]);
        zValuesAll.push_back(Z);
        auto A = applyActivation(Z, activations[layer]);
        activationsAll.push_back(A);
        current = A;
    }
    return {activationsAll, zValuesAll};
}

// Compute loss 
double NeuralNetwork::calculateLoss(const std::vector<std::vector<double>>& yTrue,
                                    const std::vector<std::vector<double>>& yPred) const {
    if (yTrue.size() != yPred.size() || (!yTrue.empty() && yTrue[0].size() != yPred[0].size()))
        throw std::invalid_argument("loss dimension mismatch");
    const std::string& outAct = activations.back();
    size_t m = yTrue.size();
    size_t k = yTrue.empty() ? 0 : yTrue[0].size();
    double loss = 0.0;
    if (outAct == "softmax") {
        // categorical cross-entropy
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < k; ++j) {
                double yp = std::min(1.0 - 1e-15, std::max(1e-15, yPred[i][j]));
                loss += -yTrue[i][j] * std::log(yp);
            }
        }
        loss /= static_cast<double>(m);
    } else if (outAct == "sigmoid") {
        // binary cross entropy
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < k; ++j) {
                double yp = std::min(1.0 - 1e-15, std::max(1e-15, yPred[i][j]));
                loss += -(yTrue[i][j] * std::log(yp) + (1.0 - yTrue[i][j]) * std::log(1.0 - yp));
            }
        }
        loss /= static_cast<double>(m);
    } else {
        // MSE
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < k; ++j) {
                double d = yTrue[i][j] - yPred[i][j];
                loss += d * d;
            }
        }
        loss /= static_cast<double>(m);
    }
    return loss;
}

// Backpropagation
void NeuralNetwork::backward(const std::vector<std::vector<double>>& X,
                             const std::vector<std::vector<double>>& y,
                             const std::vector<std::vector<std::vector<double>>>& activationsAll,
                             const std::vector<std::vector<std::vector<double>>>& zValuesAll) {
    size_t m = X.size();
    
    std::vector<std::vector<double>> currentError = subtract(activationsAll.back(), y);

    std::vector<std::vector<std::vector<double>>> weightGrads(weights.size());
    std::vector<std::vector<double>> biasGrads(weights.size());

    for (int layer = static_cast<int>(weights.size()) - 1; layer >= 0; --layer) {
        const auto& A_prev = activationsAll[layer];
        const auto& W = weights[layer];

        
        auto dW = matmul(transpose(A_prev), currentError);
        dW = scalarMat(dW, 1.0 / static_cast<double>(m));

        
        std::vector<double> db(currentError[0].size(), 0.0);
        for (const auto& row : currentError) {
            for (size_t j = 0; j < db.size(); ++j) db[j] += row[j];
        }
        for (double& v : db) v /= static_cast<double>(m);

        weightGrads[layer] = dW;
        biasGrads[layer] = db;

        if (layer > 0) {
            
            auto WT = transpose(W);
            auto prevError = matmul(currentError, WT);
            auto actDer = activationDerivativeFromActivated(activationsAll[layer], activations[layer - 1]);
            currentError = hadamard(prevError, actDer);
        }
    }

    // Update parameters
    for (size_t i = 0; i < weights.size(); ++i) {
        
        auto scaled = scalarMat(weightGrads[i], learningRate);
        weights[i] = subtract(weights[i], scaled);
       
        for (size_t j = 0; j < biases[i].size(); ++j) biases[i][j] -= learningRate * biasGrads[i][j];
    }
}

// Training loop
std::vector<double> NeuralNetwork::train(const std::vector<std::vector<double>>& X_train,
                                         const std::vector<std::vector<double>>& y_train,
                                         int epochs,
                                         bool verbose) {
    std::vector<double> lossHistory;
    lossHistory.reserve(static_cast<size_t>(epochs));
    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto fw = forward(X_train);
        const auto& activationsAll = fw.first;
        const auto& zValuesAll = fw.second;
        double loss = calculateLoss(y_train, activationsAll.back());
        lossHistory.push_back(loss);
        backward(X_train, y_train, activationsAll, zValuesAll);
        if (verbose && (epoch % 10 == 0 || epoch == epochs - 1)) {
            std::cout << "Epoch " << epoch << "/" << epochs << ", Loss: " << std::fixed << std::setprecision(6) << loss << "\n";
        }
    }
    isTrained = true;
    return lossHistory;
}

// Inference
std::vector<std::vector<double>> NeuralNetwork::predict(const std::vector<std::vector<double>>& X) const {
    if (!isTrained) {
        std::cerr << "Model not trained yet." << std::endl;
    }
    auto fw = forward(X);
    return fw.first.back();
}

// Simple CSV reader
static std::vector<std::vector<double>> readCsvHead(const std::string& filename, int expectedCols) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Failed to open file: " + filename);
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> values;
        while (std::getline(ss, cell, ',')) {
            if (!cell.empty() && (cell.back() == '\r' || cell.back() == '\n')) cell.pop_back();
            values.push_back(std::stod(cell));
        }
        if (expectedCols > 0 && static_cast<int>(values.size()) < expectedCols) {
            throw std::runtime_error("CSV row has fewer columns than expected: " + filename);
        }
        rows.push_back(std::move(values));
    }
    return rows;
}

// Load train/test CSVs
std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<std::vector<double>>>
NeuralNetwork::loadData(const std::string& trainFilename,
                        const std::string& testFilename,
                        int inputSize,
                        int outputSize) {
    if (inputSize <= 0 || outputSize <= 0) {
        throw std::invalid_argument("inputSize and outputSize must be positive");
    }
    auto trainRows = readCsvHead(trainFilename, inputSize + outputSize);
    auto testRows = readCsvHead(testFilename, inputSize + outputSize);

    auto splitXY = [&](const std::vector<std::vector<double>>& rows) {
        std::vector<std::vector<double>> X;
        std::vector<std::vector<double>> y;
        X.reserve(rows.size());
        y.reserve(rows.size());
        for (const auto& r : rows) {
            std::vector<double> xi(r.begin(), r.begin() + inputSize);
            std::vector<double> yi(r.end() - outputSize, r.end());
            X.push_back(std::move(xi));
            y.push_back(std::move(yi));
        }
        return std::make_pair(X, y);
    };

    auto trainXY = splitXY(trainRows);
    auto testXY = splitXY(testRows);
    return std::make_tuple(trainXY.first, trainXY.second, testXY.first, testXY.second);
}

// Demonstration function (AI generated)

int main() {
    using std::cout;
    using std::endl;
    cout << std::string(60, '=') << "\n";
    cout << "NEURAL NETWORK TRAINING AND TESTING" << "\n";
    cout << std::string(60, '=') << "\n";

    cout << "\n\xF0\x9F\x94\xA7 STEP 1: Initializing Neural Network\n";
    cout << std::string(40, '-') << "\n";
    std::vector<int> layers = {4, 8, 6, 3};
    std::vector<std::string> activations = {"relu", "relu", "softmax"};
    double learningRate = 0.1;
    NeuralNetwork nn(layers, activations, learningRate);
    auto totalParams = 0LL;
    for (size_t i = 0; i + 1 < layers.size(); ++i) totalParams += 1LL * layers[i] * layers[i + 1] + layers[i + 1];
    cout << "\xE2\x9C\x85 Network Architecture: [4, 8, 6, 3]\n";
    cout << "\xE2\x9C\x85 Activations: relu, relu, softmax\n";
    cout << "\xE2\x9C\x85 Learning Rate: " << learningRate << "\n";
    cout << "\xE2\x9C\x85 Total Parameters: " << totalParams << "\n";

    cout << "\n\xF0\x9F\x93\x8A STEP 2: Loading Training and Test Data\n";
    cout << std::string(40, '-') << "\n";
    std::vector<std::vector<double>> X_train, y_train, X_test, y_test;
    try {
        auto data = NeuralNetwork::loadData("Python/train_data_nn.csv", "Python/test_data_nn.csv", layers.front(), layers.back());
        X_train = std::get<0>(data);
        y_train = std::get<1>(data);
        X_test = std::get<2>(data);
        y_test = std::get<3>(data);
        cout << "\xE2\x9C\x85 Training data loaded: " << X_train.size() << " samples\n";
        cout << "\xE2\x9C\x85 Test data loaded: " << X_test.size() << " samples\n";
        cout << "\xE2\x9C\x85 Input features: " << X_train[0].size() << "\n";
        cout << "\xE2\x9C\x85 Output classes: " << y_train[0].size() << "\n";
    } catch (const std::exception& e) {
        cout << "\xE2\x9D\x8C Error loading data: " << e.what() << "\n";
        cout << "\xF0\x9F\x93\x9D Make sure train_data_nn.csv and test_data_nn.csv exist in the Python/ directory\n";
        return 0;
    }

    cout << "\n\xF0\x9F\x94\xAE STEP 3: Testing Forward Pass (Before Training)\n";
    cout << std::string(40, '-') << "\n";
    auto prePred = nn.predict(std::vector<std::vector<double>>(X_test.begin(), X_test.begin() + std::min<size_t>(3, X_test.size())));
    cout << "\xE2\x9C\x85 Sample predictions (untrained network):\n";
    for (size_t i = 0; i < prePred.size(); ++i) {
        const auto& p = prePred[i];
        cout << "   Sample " << (i + 1) << ": [" << std::fixed << std::setprecision(3) << p[0] << ", " << p[1] << ", " << p[2] << "]\n";
    }

    cout << "\n\xF0\x9F\x8E\xAF STEP 4: Training Neural Network\n";
    cout << std::string(40, '-') << "\n";
    cout << "Starting training process...\n";
    int epochs = 50;
    auto lossHistory = nn.train(X_train, y_train, epochs, true);
    cout << "\n\xE2\x9C\x85 Training completed!\n";
    cout << std::fixed << std::setprecision(6);
    cout << "\xE2\x9C\x85 Initial Loss: " << lossHistory.front() << "\n";
    cout << "\xE2\x9C\x85 Final Loss: " << lossHistory.back() << "\n";
    cout << "\xE2\x9C\x85 Loss Reduction: " << ((lossHistory.front() - lossHistory.back()) / lossHistory.front() * 100.0) << "%\n";

    cout << "\n\xF0\x9F\xA7\xAA STEP 5: Testing Predictions (After Training)\n";
    cout << std::string(40, '-') << "\n";
    auto predictions = nn.predict(X_test);
    cout << "\xE2\x9C\x85 Sample predictions (trained network):\n";
    for (size_t i = 0; i < predictions.size(); ++i) {
        const auto& pred = predictions[i];
        const auto& actual = y_test[i];
        size_t predictedClass = std::distance(pred.begin(), std::max_element(pred.begin(), pred.end()));
        size_t actualClass = std::distance(actual.begin(), std::max_element(actual.begin(), actual.end()));
        double confidence = pred[predictedClass];
        cout << "   Sample " << (i + 1) << ":\n";
        cout << "      Predicted: Class " << predictedClass << " (confidence: " << std::setprecision(3) << confidence << ")\n";
        cout << "      Actual:    Class " << actualClass << "\n";
        cout << "      Correct:   " << (predictedClass == actualClass ? "\xE2\x9C\x85 Yes" : "\xE2\x9D\x8C No") << "\n";
    }

    cout << "\n\xF0\x9F\x93\x88 STEP 6: Final Performance Metrics\n";
    cout << std::string(40, '-') << "\n";
    size_t correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        size_t pc = std::distance(predictions[i].begin(), std::max_element(predictions[i].begin(), predictions[i].end()));
        size_t ac = std::distance(y_test[i].begin(), std::max_element(y_test[i].begin(), y_test[i].end()));
        if (pc == ac) ++correct;
    }
    double accuracy = predictions.empty() ? 0.0 : static_cast<double>(correct) / static_cast<double>(predictions.size());
    cout << std::fixed << std::setprecision(4);
    cout << "\xE2\x9C\x85 Test Accuracy: " << accuracy << " (" << (accuracy * 100.0) << "%)\n";
    cout << "\xE2\x9C\x85 Correct Predictions: " << correct << "/" << predictions.size() << "\n";

    cout << "\n\xF0\x9F\x8E\x89 STEP 7: Training Summary\n";
    cout << std::string(40, '-') << "\n";
    cout << "\xE2\x9C\x85 Network successfully trained with " << 50 << " epochs\n";
    cout << "\xE2\x9C\x85 Final test accuracy: " << std::setprecision(2) << (accuracy * 100.0) << "%\n";
    cout << "\xE2\x9C\x85 Model is ready for use!\n";

    cout << "\n" << std::string(60, '=') << "\n";
    cout << "NEURAL NETWORK DEMO COMPLETED SUCCESSFULLY!\n";
    cout << std::string(60, '=') << "\n";
    return 0;
}



# GNNs-From-Scratch

**Graph Neural Networks and Machine Learning Algorithms implemented from scratch in CUDA**

This repository showcases advanced implementations of classical Machine Learning algorithms, Neural Networks, and Graph Neural Networks built entirely from scratch using CUDA for GPU acceleration. The CUDA and C++ implementations demonstrate real-world high-performance computing applications, while Python serves as a reference implementation for understanding the underlying mathematics and logic.

Read this to learn more about this project : https://medium.com/@nehalsinghal77/learning-gnns-05dab59fccc3 

Project Status : Completed.

## Project Goals

- **Primary Focus**: Master GPU programming with CUDA for ML acceleration
- **Performance**: Achieve significant speedup over CPU implementations locally
- **Learning**: Build production-ready ML algorithms from first principles
- **Learning**: Master C++ and CUDA programming for high-performance computing
- **Learning**: Understand Graph Neural Networks and their GPU implementation strategies
- **Learning**: Gain expertise in GPU programming and optimization techniques

## Architecture

### **CUDA**
GPU-accelerated implementations optimized for real-world applications:

```
CUDA/
├── NeuralNetwork.cu        # Multi-layer neural networks with backpropagation
├── NeuralNetwork.h         # Header with activation functions (ReLU, Sigmoid, Tanh, Softmax)
├── GNN.cu                  # Graph Neural Networks with message passing
├── GNN.h                   # Graph convolution and aggregation algorithms
├── LinearRegression.cu     # Gradient descent optimization on GPU
├── LinearRegression.h      # Minimal dependencies, maximum performance
└── gnn_demo.cu               # Graph learning demonstration
├── Common/
│   ├── MatrixOps.cu         # Core matrix operations (GEMM, transposition, reductions)
│   ├── MatrixOps.h          # High-performance linear algebra kernels
│   ├── Utils.cu             # CUDA debugging, profiling, and validation utilities
│   └── Utils.h              # Helper functions for data generation and evaluation
```

### **Python**
Educational implementations focusing on clarity and mathematical understanding:

```
Python/
├── LinearRegression.py      # Clean gradient descent implementation
├── NeuralNetwork.py         # Multi-layer perceptron with backprop
├── GraphAlgorithms.py       # Classic graph algorithms (Dijkstra, Kruskal, Boruvka)
├── GNN.py                   # Message-passing neural networks
├── RNN.py                   # Recurrent neural networks (LSTM/GRU inspired)
└── *.csv                    # Sample datasets for training
```

### **C++**
CPU implementations for performance comparison:

```
C++/
├── LinearRegression.cpp     # Vectorized gradient descent
├── NeuralNetwork.cpp        # Optimized C++ neural networks
├── NeuralNetwork.h         
├── GraphAlgorithms.cpp      # Efficient graph algorithms
└── GraphAlgorithms.h      
```

## Getting Started

### **Google Colab Setup**
```python
# Complete setup for Google Colab GPU
!git clone https://github.com/yourusername/GNNs-From-Scratch.git
%cd GNNs-From-Scratch/CUDA

# Install build tools
!apt-get update -qq && apt-get install -y build-essential -qq

# Update Makefile for Colab GPU (T4 uses sm_75)
!sed -i 's/-arch=sm_50/-arch=sm_75/g' Makefile

# Compile and run
!make clean && make all
!./gnn_demo
```

## Key Features

### **Neural Networks**
- **GPU Acceleration**: Faster than CPU implementations
- **Multiple Activations**: ReLU, Sigmoid, Tanh, Leaky ReLU, Softmax
- **Advanced Training**: Backpropagation with momentum and regularization
- **Batch Processing**: Efficient parallel computation across samples

### **Graph Neural Networks**
- **Node-Based Tasks**: Classification, Regression, Property Prediction
- **Message Passing**: Sophisticated neighbor aggregation strategies
- **Graph Convolution**: Custom kernels for graph convolutional layers
- **GPU Acceleration**: 10-100x faster than CPU implementations
- **Scalable**: Handle large graphs with millions of nodes
- **Memory Efficient**: Optimized GPU memory management
- **Task-Specific**: Different activations and loss functions per task type

### **Linear Regression**
- **Gradient Descent**: CUDA-accelerated optimization
- **Mini-batching**: Parallel gradient computation
- **Real-time**: Sub-millisecond prediction times

## GNN Task Types

### **Node Classification**
- **Purpose**: Classify nodes into discrete categories
- **Example**: Social media user classification (influencer, regular, bot)
- **Output**: Softmax probabilities over classes
- **Loss**: Cross-entropy loss
- **Use Case**: Fraud detection, user profiling, content moderation

### **Node Regression** 
- **Purpose**: Predict continuous values for nodes
- **Example**: Recommendation system rating prediction
- **Output**: Linear activation (unbounded)
- **Loss**: Mean squared error
- **Use Case**: Price prediction, popularity scoring, influence estimation

### **Property Prediction**
- **Purpose**: Predict binary node attributes
- **Example**: Missing node property inference
- **Output**: Sigmoid activation (0-1 bounded)
- **Loss**: Binary cross-entropy
- **Use Case**: Link prediction, anomaly detection, feature completion


## Learning Path

1. **Start with Python** → Revise mathematical concepts, learn GNNs
2. **Study C++ implementations** → Learn CPU optimization techniques and C++
3. **Dive into CUDA** → Learn GPU programming, CUDA and parallel algorithms
4. **Experiment** → Modify parameters, architectures, and optimization strategies. Test on Google Collab GPUs.
5. **Benchmark** → Compare performance across different implementations

## Future Roadmap

### **FPGA Implementation via High-Level Synthesis**
This project serves as a foundation for understanding how machine learning algorithms map to hardware implementations through the High-Level Synthesis (HLS) pipelines and understanding why different hardware is optimal for different models. 

- **FPGA Implementation**: SystemVerilog/VHDL ports for ultra-low latency inference
- **HLS Pipeline Understanding**: Exploring the C++ to Verilog/VHDL translation process for FPGA deployment
- **RTL Understanding**: Deep dive into how ML algorithms map to Register Transfer Level representations
- **Custom ML Accelerators**: FPGA-based implementations of Machine Learning Algorithms.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

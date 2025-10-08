# CUDA ML Framework Makefile

# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -std=c++14 -O3 -arch=sm_50 -lineinfo
INCLUDES = -I. -I./Common
LIBS = -lcudart

# Directories
SRC_DIR = .
COMMON_DIR = ./Common
EXAMPLES_DIR = ./Examples
BUILD_DIR = ./build

# Source files
COMMON_SOURCES = $(COMMON_DIR)/MatrixOps.cu $(COMMON_DIR)/Utils.cu
ML_SOURCES = $(SRC_DIR)/LinearRegression.cu $(SRC_DIR)/NeuralNetwork.cu $(SRC_DIR)/GNN.cu
EXAMPLE_SOURCES = $(EXAMPLES_DIR)/linear_regression_demo.cu $(EXAMPLES_DIR)/neural_network_demo.cu $(EXAMPLES_DIR)/gnn_demo.cu

# Object files
COMMON_OBJECTS = $(COMMON_SOURCES:.cu=.o)
ML_OBJECTS = $(ML_SOURCES:.cu=.o)
EXAMPLE_OBJECTS = $(EXAMPLE_SOURCES:.cu=.o)

# Executables
EXECUTABLES = linear_regression_demo neural_network_demo gnn_demo

# Default target
all: $(EXECUTABLES)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile common utilities
$(COMMON_DIR)/%.o: $(COMMON_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Compile ML algorithms
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Compile examples
$(EXAMPLES_DIR)/%.o: $(EXAMPLES_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Link executables
linear_regression_demo: $(EXAMPLES_DIR)/linear_regression_demo.o $(SRC_DIR)/LinearRegression.o $(COMMON_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

neural_network_demo: $(EXAMPLES_DIR)/neural_network_demo.o $(SRC_DIR)/NeuralNetwork.o $(COMMON_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

gnn_demo: $(EXAMPLES_DIR)/gnn_demo.o $(SRC_DIR)/GNN.o $(COMMON_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

# Clean build artifacts
clean:
	rm -f $(COMMON_OBJECTS) $(ML_OBJECTS) $(EXAMPLE_OBJECTS) $(EXECUTABLES)
	rm -rf $(BUILD_DIR)

# Run examples
run-linear: linear_regression_demo
	./linear_regression_demo

run-neural: neural_network_demo
	./neural_network_demo

run-gnn: gnn_demo
	./gnn_demo

# Run all examples
run-all: run-linear run-neural run-gnn

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	@nvcc --version || echo "CUDA not found!"
	@nvidia-smi || echo "NVIDIA driver not found!"

# Help
help:
	@echo "Available targets:"
	@echo "  all          - Build all executables"
	@echo "  clean        - Remove build artifacts"
	@echo "  run-linear   - Run linear regression demo"
	@echo "  run-neural   - Run neural network demo"
	@echo "  run-gnn      - Run GNN demo"
	@echo "  run-all      - Run all demos"
	@echo "  check-cuda   - Check CUDA installation"
	@echo "  help         - Show this help"

.PHONY: all clean run-linear run-neural run-gnn run-all check-cuda help






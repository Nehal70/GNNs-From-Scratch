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
DEMO_SOURCES = $(SRC_DIR)/gnn_demo.cu

# Object files
COMMON_OBJECTS = $(COMMON_SOURCES:.cu=.o)
ML_OBJECTS = $(ML_SOURCES:.cu=.o)
DEMO_OBJECTS = $(DEMO_SOURCES:.cu=.o)

# Executables
EXECUTABLES = gnn_demo

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

# Compile demos
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Link executables
gnn_demo: $(SRC_DIR)/gnn_demo.o $(SRC_DIR)/GNN.o $(COMMON_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

# Clean build artifacts
clean:
	rm -f $(COMMON_OBJECTS) $(ML_OBJECTS) $(DEMO_OBJECTS) $(EXECUTABLES)
	rm -rf $(BUILD_DIR)

# Run demo
run-gnn: gnn_demo
	./gnn_demo

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	@nvcc --version || echo "CUDA not found!"
	@nvidia-smi || echo "NVIDIA driver not found!"

# Help
help:
	@echo "Available targets:"
	@echo "  all          - Build gnn demo"
	@echo "  clean        - Remove build artifacts"
	@echo "  run-gnn      - Run GNN demo"
	@echo "  check-cuda   - Check CUDA installation"
	@echo "  help         - Show this help"

.PHONY: all clean run-gnn check-cuda help






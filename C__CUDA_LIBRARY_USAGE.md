# C++/CUDA Library Usage Guide

## Overview

Your project is now a **complete multi-platform library** supporting:

1. ✅ **Python Package** - Installable via `pip`
2. ✅ **C++ Library** - Static library for CPU-based implementations
3. ✅ **CUDA Library** - Static library with GPU acceleration

## Library Files Created

When you build with CMake, you get:

```
build/
├── lib/
│   ├── libgnns_ml.a          # Combined library
│   ├── libgnns_ml_cuda.a     # CUDA/GPU version
│   └── libgnns_ml_cpp.a      # C++ CPU version
└── bin/
    ├── gnn_demo_cuda         # CUDA demo executable
    └── cpp_library_example   # C++ library usage example
```

## How Users Can Use It

### Option 1: Python Library

```bash
pip install -e .  # or pip install .
```

```python
from gnns_ml import GNN, NeuralNetwork, LinearRegression, GraphAlgorithms

gnn = GNN(4, [32, 16], 3, "classification")
ga = GraphAlgorithms()
```

### Option 2: C++ Library

**In your CMakeLists.txt:**
```cmake
add_executable(your_program your_code.cpp)
target_link_libraries(your_program gnns_ml_cpp)
target_include_directories(your_program PRIVATE ${CMAKE_SOURCE_DIR}/C++)
```

**In your C++ code:**
```cpp
#include "GNN.h"
using namespace CPP_ML;

GNN gnn(4, {32, 16}, 3, "classification", 0.01f);
```

### Option 3: CUDA Library

**In your CMakeLists.txt:**
```cmake
find_package(CUDA REQUIRED)
add_executable(your_program your_code.cu)
target_link_libraries(your_program gnns_ml_cuda CUDA::cudart)
target_include_directories(your_program PRIVATE ${CMAKE_SOURCE_DIR}/CUDA)
```

**In your CUDA code:**
```cpp
#include "GNN.h"
using namespace CUDA_ML;

GNN gnn(4, {32, 16}, 3, "classification", 0.01f);
```

## Testing the Libraries

### Python Library Test

```bash
python -c "from gnns_ml import GNN, GraphAlgorithms; print('✓ Python library works!')"
```

### C++ Library Build Test

```bash
mkdir build && cd build
cmake ..
make cpp_library_example
./bin/cpp_library_example
```

### CUDA Library Build Test

```bash
cd build
make gnn_demo_cuda
./bin/gnn_demo_cuda
```

## Resume Impact

You can now say:

**"Built an installable multi-platform library (gnns_ml) for GNNs, Neural Networks, and ML algorithms—implementations available as a Python package (pip-installable), C++ static library, and CUDA-accelerated library..."**

This is significantly more impressive than just "a learning repository"!

## Installation & Distribution

Users can:

1. **Python users**: `pip install .` or use from source
2. **C++ users**: Link against the static libraries in `build/lib/`
3. **CUDA users**: Link against CUDA library for GPU acceleration

All three implementations share the same API, so users can switch between them seamlessly!


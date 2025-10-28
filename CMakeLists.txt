cmake_minimum_required(VERSION 3.18)
project(GNNS_ML VERSION 0.1.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA settings
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_EXAMPLES "Build example programs" ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/CUDA)
include_directories(${CMAKE_SOURCE_DIR}/CUDA/Common)
include_directories(${CMAKE_SOURCE_DIR}/C++)

# Source files
set(CUDA_SOURCES
    CUDA/Common/MatrixOps.cu
    CUDA/Common/Utils.cu
    CUDA/LinearRegression.cu
    CUDA/NeuralNetwork.cu
    CUDA/GNN.cu
)

set(CPP_SOURCES
    C++/LinearRegression.cpp
    C++/NeuralNetwork.cpp
    C++/GNN.cpp
    C++/GraphAlgorithms.cpp
)

# Create CUDA library (GPU-accelerated)
add_library(gnns_ml_cuda STATIC ${CUDA_SOURCES})
set_property(TARGET gnns_ml_cuda PROPERTY CUDA_ARCHITECTURES "50;75;80")
target_link_libraries(gnns_ml_cuda PUBLIC CUDA::cudart)

# Create C++ library (CPU-only)
add_library(gnns_ml_cpp STATIC ${CPP_SOURCES})

# Create combined library (links both)
add_library(gnns_ml STATIC ${CUDA_SOURCES} ${CPP_SOURCES})
set_property(TARGET gnns_ml PROPERTY CUDA_ARCHITECTURES "50;75;80")
target_link_libraries(gnns_ml PUBLIC CUDA::cudart)

# Set output directories
set_target_properties(gnns_ml PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Install targets
install(TARGETS gnns_ml gnns_ml_cuda gnns_ml_cpp
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install headers
install(FILES 
    CUDA/GNN.h
    CUDA/NeuralNetwork.h
    CUDA/LinearRegression.h
    CUDA/Common/MatrixOps.h
    CUDA/Common/Utils.h
    C++/GNN.h
    C++/NeuralNetwork.h
    C++/LinearRegression.h
    C++/GraphAlgorithms.h
    DESTINATION include/gnns_ml
)

# Build examples if enabled
if(BUILD_EXAMPLES)
    # CUDA demo
    add_executable(gnn_demo_cuda CUDA/gnn_demo.cu)
    target_link_libraries(gnn_demo_cuda gnns_ml_cuda)
    
    # C++ library usage example
    add_executable(cpp_library_example examples/cpp_library_example.cpp)
    target_link_libraries(cpp_library_example gnns_ml_cpp)
    
    # Also install examples
    install(TARGETS gnn_demo_cuda cpp_library_example DESTINATION bin)
endif()


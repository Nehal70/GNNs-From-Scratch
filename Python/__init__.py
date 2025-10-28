"""
GNNs-From-Scratch: From-scratch implementations of ML algorithms

A collection of implementations including:
- Graph Neural Networks (GNNs)
- Neural Networks
- Linear Regression
- Graph Algorithms

Usage:
    >>> from gnns_ml import GNN, NeuralNetwork, LinearRegression
    >>> import numpy as np
    >>> 
    >>> # Create a GNN for node classification
    >>> gnn = GNN(input_dim=4, hidden_dims=[64, 32], output_dim=3, 
    ...            task_type="classification")
"""

__version__ = "0.1.0"
__all__ = ["GNN", "NeuralNetwork", "LinearRegression", "GraphAlgorithms"]


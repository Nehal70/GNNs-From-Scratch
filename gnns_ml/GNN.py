import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Union
import random
import math

#Was recommended to implement this with a Layer class inside the GNN class.

class GraphConvolutionLayer:
    """
    single graph convolution layer that does message passing between connected nodes.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        """
        initializing weights and bias for this layer.
        """
        # weight matrix for transforming node features
        self.w = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        # bias vector for each output dimension
        self.b = np.zeros(output_dim)
        
    def forward(self, node_features: np.ndarray, adjacency_matrix: np.ndarray) -> np.ndarray:
        """
        forward pass through graph convolution layer.
        aggregates neighbor information and applies linear transformation.
        """
        # step 1: aggregate neighbor features by summing them up
        # adjacency matrix tells us which nodes are connected
        neighbor_info = adjacency_matrix @ node_features
        
        # step 2: apply linear transformation (weight * features + bias)
        transformed_features = neighbor_info @ self.w + self.b
        
        return transformed_features

class GNN:
    """
    graph neural network that can handle multiple node-based tasks.
    uses message passing to learn node representations from graph structure.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                 task_type: str = "classification", learning_rate: float = 0.01):
        """
        initializing gnn with specified architecture for different tasks.
        self explanatory variable names
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.task_type = task_type
        self.lr = learning_rate
        self.is_trained = False
        
        # build the gnn layers
        self.gnn_layers = []
        
        # first layer
        current_dim = input_dim

        for hidden_size in hidden_dims:
            conv_layer = GraphConvolutionLayer(current_dim, hidden_size)
            self.gnn_layers.append(conv_layer)
            current_dim = hidden_size
            
        # final layer
        self.final_layer = GraphConvolutionLayer(current_dim, output_dim)
        
    def _apply_activation(self, x: np.ndarray, activation: str) -> np.ndarray:
        """
        apply activation function to layer outputs.
        """
        if activation == "relu":
            return np.maximum(0, x)
        elif activation == "sigmoid":
            # clip to prevent overflow
            x = np.clip(x, -500, 500)
            return 1 / (1 + np.exp(-x))
        elif activation == "tanh":
            return np.tanh(x)
        elif activation == "softmax":
            # for classification
            x_shifted = x - np.max(x, axis=1, keepdims=True)
            exp_x = np.exp(x_shifted)
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:
            # linear activation for regression
            return x  
    
    def forward(self, node_features: np.ndarray, adjacency_matrix: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        forward pass through entire gnn.
        processes node features through all graph convolution layers.
        returns both final output and intermediate activations for gradient computation.
        """
        h = node_features
        activations = [h]  # store all intermediate activations
        
        # pass through hidden layers with relu activation
        for conv_layer in self.gnn_layers:
            h = conv_layer.forward(h, adjacency_matrix)
            h = self._apply_activation(h, "relu")
            activations.append(h)
            
        # final output layer
        final_output = self.final_layer.forward(h, adjacency_matrix)
        
        # apply final activation on output based on task type
        if self.task_type == "classification":
            final_output = self._apply_activation(final_output, "softmax")
        elif self.task_type == "regression":
            final_output = self._apply_activation(final_output, "linear")  # no activation
        elif self.task_type == "property_prediction":
            final_output = self._apply_activation(final_output, "sigmoid")  # bounded output
        
        activations.append(final_output)
        return final_output, activations
    
    def _compute_graph_gradients(self, node_features: np.ndarray, adjacency_matrix: np.ndarray, 
                                labels: np.ndarray, activations: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        compute gradients properly for gnn using graphical backpropagation that accounts for how gradients flow through the graph structure.
        """
        # compute output layer gradient
        output_grad = activations[-1] - labels
        
        # store gradients for each layer
        weight_grads = []
        bias_grads = []
        
        # backprop through final layer
        final_layer_grad_w = activations[-2].T @ output_grad
        final_layer_grad_b = np.mean(output_grad, axis=0)
        weight_grads.append(final_layer_grad_w)
        bias_grads.append(final_layer_grad_b)
        
        # compute gradient flowing back through final layer
        current_grad = output_grad @ self.final_layer.w.T
        
        # backprop through hidden layers with graph structure
        for i in reversed(range(len(self.gnn_layers))):
            # apply activation derivative for current layer
            current_grad *= self._relu_derivative(activations[i+1])
            
            # compute gradients for current layer weights
            if i == 0:
                # first hidden layer uses input features
                layer_input = node_features
            else:
                layer_input = activations[i]
            
            # weight gradient: dW = input.T @ gradient
            dW = layer_input.T @ current_grad
            # bias gradient: db = mean(gradient)
            db = np.mean(current_grad, axis=0)
            
            weight_grads.insert(0, dW)
            bias_grads.insert(0, db)
            
            # propagate gradient backward through graph convolution
            if i > 0:
                # gradient flows back through the graph convolution operation
                # this is the key difference from regular neural networks
                current_grad = self._backprop_graph_conv(current_grad, adjacency_matrix, self.gnn_layers[i].w)
        
        return weight_grads, bias_grads
    
    def _backprop_graph_conv(self, grad_output: np.ndarray, adjacency_matrix: np.ndarray, 
                            weight_matrix: np.ndarray) -> np.ndarray:
        """
        backpropagate gradient through graph convolution operation.
        """
        # gradient flows back through the linear transformation
        grad_linear = grad_output @ weight_matrix.T
        
        # gradient flows back through the adjacency matrix multiplication
        grad_input = adjacency_matrix.T @ grad_linear
        
        return grad_input
    
    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        compute derivative of relu activation function.
        """
        return np.where(x > 0, 1, 0)

    def _calculate_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        calculate loss based on task type.
        follow the examples to understand what each task type does in a GNN.
        """
        if self.task_type == "classification":
            # cross entropy loss for classification tasks
            # example: social media network calculates loss here to see how well it predicts user types
            y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
            return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
            
        elif self.task_type == "regression":
            # mean squared error for regression tasks
            # example: recommendation system calculates loss here to see how well it predicts user ratings
            return np.mean((y_true - y_pred) ** 2)
            
        elif self.task_type == "property_prediction":
            # binary cross entropy for property prediction tasks
            # example: fraud detection system calculates loss here to see how well it predicts suspicious behavior
            y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
            return -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
    
    def train(self, node_features: np.ndarray, adjacency_matrix: np.ndarray, 
              labels: np.ndarray, epochs: int = 100, verbose: bool = True) -> List[float]:
        """
        train the gnn using proper graph-aware backpropagation.
        gradients flow through the graph structure, not just the neural network.
        """
        loss_tracker = []
        
        for epoch in range(epochs):
            # forward pass - get both predictions and activations
            preds, activations = self.forward(node_features, adjacency_matrix)
            
            # calculate loss
            current_loss = self._calculate_loss(labels, preds)
            loss_tracker.append(current_loss)
            
            # compute gradients using graph-aware backpropagation
            weight_grads, bias_grads = self._compute_graph_gradients(node_features, adjacency_matrix, labels, activations)
            
            # update weights and biases using computed gradients
            for i, conv_layer in enumerate(self.gnn_layers):
                conv_layer.w -= self.lr * weight_grads[i]
                conv_layer.b -= self.lr * bias_grads[i]
            
            # update final layer
            self.final_layer.w -= self.lr * weight_grads[-1]
            self.final_layer.b -= self.lr * bias_grads[-1]
            
            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                print(f"epoch {epoch}/{epochs}, loss: {current_loss:.6f}")
        
        self.is_trained = True
        return loss_tracker
    
    def predict(self, node_features: np.ndarray, adjacency_matrix: np.ndarray) -> np.ndarray:
        """
        make predictions on new data.
        """
        if not self.is_trained:
            print("model not trained yet.")
            return None
            
        preds, _ = self.forward(node_features, adjacency_matrix)
        return preds

def main():
    """
    Documentation and usage guide for the GNN class (AI-generated documentation).
    """
    print("=" * 80)
    print("GRAPH NEURAL NETWORK (GNN) CLASS - USAGE DOCUMENTATION")
    print("=" * 80)
    
    print("""
    OVERVIEW:
    --------
    This GNN implementation supports three types of node-based tasks:
    1. Node Classification - Classify nodes into different categories
    2. Node Regression - Predict continuous values for nodes  
    3. Property Prediction - Predict missing node attributes
    
    DATA REQUIREMENTS:
    -----------------
    You need two main inputs for your graph data:
    
    1. Node Features: NumPy array of shape [num_nodes, num_features]
       - Each row represents one node's feature vector
       - Example: User features like [age, followers, posts, activity_score]
       - Can be loaded from CSV, database, or generated programmatically
    
    2. Adjacency Matrix: NumPy array of shape [num_nodes, num_nodes] 
       - Binary matrix where 1 indicates connection between nodes, 0 means no connection
       - Should be symmetric for undirected graphs (adjacency_matrix[i,j] = adjacency_matrix[j,i])
       - Diagonal should be 0 (no self-loops)
       - Example: Social network connections, citation networks, molecular bonds
    
    INSTANTIATING A GNN:
    -------------------
    gnn = GNN(
        input_dim=num_features,        # Number of input features per node
        hidden_dims=[64, 32],         # List of hidden layer sizes
        output_dim=num_classes,       # Output size (classes for classification, 1 for regression)
        task_type="classification",    # "classification", "regression", or "property_prediction"
        learning_rate=0.01            # Learning rate for gradient descent
    )
    
    TASK-SPECIFIC CONFIGURATION:
    ---------------------------
    
    FOR NODE CLASSIFICATION:
    - task_type="classification"
    - output_dim = Number of classes (e.g., 3 for "influencer", "regular", "bot")
    - Labels should be one-hot encoded: np.eye(num_classes)[class_indices]
    - Loss function: Cross-entropy
    - Output activation: Softmax
    
    FOR NODE REGRESSION:
    - task_type="regression" 
    - output_dim=1
    - Labels should be continuous values: np.array([[score1], [score2], ...])
    - Loss function: Mean squared error
    - Output activation: Linear (no activation)
    
    FOR PROPERTY PREDICTION:
    - task_type="property_prediction"
    - output_dim=1 (for binary properties)
    - Labels should be binary: np.array([[0], [1], [0], ...])
    - Loss function: Binary cross-entropy
    - Output activation: Sigmoid
    
    TRAINING YOUR GNN:
    -----------------
    loss_history = gnn.train(
        node_features=your_node_features,    # Shape: [num_nodes, num_features]
        adjacency_matrix=your_adjacency,     # Shape: [num_nodes, num_nodes]
        labels=your_labels,                  # Shape depends on task type
        epochs=100,                          # Number of training epochs
        verbose=True                         # Print training progress
    )
    
    MAKING PREDICTIONS:
    ------------------
    predictions = gnn.predict(
        node_features=test_node_features,    # Shape: [num_nodes, num_features]
        adjacency_matrix=test_adjacency      # Shape: [num_nodes, num_nodes]
    )
    
    EXAMPLE WORKFLOW:
    ---------------
    1. Load your graph data (node features and adjacency matrix)
    2. Prepare labels according to your task type
    3. Instantiate GNN with appropriate parameters
    4. Train the model using gnn.train()
    5. Make predictions using gnn.predict()
    6. Evaluate performance using appropriate metrics
    
    PERFORMANCE TIPS:
    ---------------
    - Start with smaller hidden_dims like [32, 16] for faster training
    - Use learning_rate between 0.001 and 0.1
    - Normalize your node features for better convergence
    - Ensure your adjacency matrix is properly formatted (symmetric, no self-loops)
    - Monitor loss during training to detect overfitting
    
    ARCHITECTURE NOTES:
    ------------------
    - Uses graph convolution layers with message passing
    - Implements proper graph-aware backpropagation
    - Supports multiple hidden layers with ReLU activation
    - Final layer activation depends on task type
    - Gradients flow through graph structure during training
    """)
    
    print("\n" + "=" * 80)
    print("GNN CLASS READY FOR USE!")
    print("=" * 80)

if __name__ == "__main__":
    # set random seed for reproducible results
    np.random.seed(42)
    main()
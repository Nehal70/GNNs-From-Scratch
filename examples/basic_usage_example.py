"""
Basic Usage Examples for gnns_ml Library

This example demonstrates how to use the gnns_ml library after installation.
Run: python basic_usage_example.py
"""

import numpy as np
from gnns_ml import GNN, NeuralNetwork, LinearRegression, GraphAlgorithms

def example_gnn():
    """Example: Using Graph Neural Networks for node classification"""
    print("=" * 60)
    print("Example 1: Graph Neural Network for Node Classification")
    print("=" * 60)
    
    # Create synthetic graph data
    num_nodes = 50
    feature_dim = 4
    num_classes = 3
    
    # Random node features
    node_features = np.random.randn(num_nodes, feature_dim)
    
    # Create random adjacency matrix (undirected graph)
    adjacency_matrix = np.random.rand(num_nodes, num_nodes)
    adjacency_matrix = (adjacency_matrix > 0.3).astype(float)
    np.fill_diagonal(adjacency_matrix, 0)  # No self-loops
    
    # Create random one-hot labels
    labels = np.eye(num_classes)[np.random.randint(0, num_classes, num_nodes)]
    
    # Create and train GNN
    gnn = GNN(
        input_dim=feature_dim,
        hidden_dims=[32, 16],
        output_dim=num_classes,
        task_type="classification",
        learning_rate=0.01
    )
    
    print(f"Training GNN on {num_nodes} nodes...")
    gnn.train(node_features, adjacency_matrix, labels, epochs=50, verbose=True)
    
    # Make predictions
    predictions = gnn.predict(node_features, adjacency_matrix)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels, axis=1)
    
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"\nNode Classification Accuracy: {accuracy:.2%}")
    print()


def example_neural_network():
    """Example: Using Neural Networks for classification"""
    print("=" * 60)
    print("Example 2: Neural Network for Classification")
    print("=" * 60)
    
    # Create synthetic data
    n_samples = 200
    n_features = 10
    n_classes = 2
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    y_one_hot = np.eye(n_classes)[y]
    
    # Create and train neural network
    nn = NeuralNetwork(
        input_dim=n_features,
        hidden_dims=[32, 16],
        output_dim=n_classes,
        learning_rate=0.01
    )
    
    print(f"Training Neural Network on {n_samples} samples...")
    nn.train(X, y_one_hot, epochs=100, verbose=True)
    
    # Make predictions
    predictions = nn.predict(X)
    predicted_classes = np.argmax(predictions, axis=1)
    
    accuracy = np.mean(predicted_classes == y)
    print(f"\nNeural Network Accuracy: {accuracy:.2%}")
    print()


def example_linear_regression():
    """Example: Using Linear Regression"""
    print("=" * 60)
    print("Example 3: Linear Regression")
    print("=" * 60)
    
    # Create synthetic data (y = 2x + 1 + noise)
    n_samples = 1000
    X = np.random.randn(n_samples, 1)
    y = 2 * X + 1 + np.random.randn(n_samples, 1) * 0.1
    
    # Create and train linear regression model
    model = LinearRegression(learning_rate=0.01)
    
    print(f"Training Linear Regression on {n_samples} samples...")
    model.train(X, y, epochs=1000, verbose=True)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Calculate MSE
    mse = np.mean((y - predictions) ** 2)
    print(f"\nMean Squared Error: {mse:.6f}")
    print(f"Learned weight: {model.weights[0][0]:.4f} (expected ~2.0)")
    print(f"Learned bias: {model.bias[0]:.4f} (expected ~1.0)")
    print()


def example_graph_algorithms():
    """Example: Using Graph Algorithms"""
    print("=" * 60)
    print("Example 4: Classic Graph Algorithms")
    print("=" * 60)
    
    # Create a simple graph
    graph = [
        [0, 1, 1, 0, 0],  # Node 0 connects to 1, 2
        [1, 0, 1, 1, 0],  # Node 1 connects to 0, 2, 3  
        [1, 1, 0, 1, 1],  # Node 2 connects to 0, 1, 3, 4
        [0, 1, 1, 0, 1],  # Node 3 connects to 1, 2, 4
        [0, 0, 1, 1, 0]   # Node 4 connects to 2, 3
    ]
    
    ga = GraphAlgorithms()
    
    print(f"Running DFS on {len(graph)} nodes...")
    previsited, postvisited = ga.depth_first_search(graph)
    print(f"DFS completed. Previsit times: {previsited}")
    
    print(f"\nRunning BFS from node 0...")
    distances = ga.breadth_first_search(graph, start=0)
    print(f"BFS distances from node 0: {distances}")
    
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("GNNs-From-Scratch Library - Usage Examples")
    print("=" * 60 + "\n")
    
    print("These examples demonstrate the basic usage of the gnns_ml library.")
    print("All implementations are done from scratch for educational purposes.\n")
    
    try:
        example_linear_regression()
        example_neural_network()
        example_gnn()
        example_graph_algorithms()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you've installed the library: pip install -e .")
        raise


if __name__ == "__main__":
    main()


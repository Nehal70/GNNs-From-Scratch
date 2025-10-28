import numpy as np
import pandas as pd
from typing import Tuple, Union, List, Optional, Dict, Any
import random
import math

class NeuralNetwork:

    def __init__(self, layers: List[int], activations: List[str], learning_rate: float = 0.01):
        """
        Initializing neural network with neccesary properties.
        """

        #self explanatory variables
        self.layers = layers
        self.activations = activations
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        self.is_trained = False

        #Initialise weights and biases between layers
        for i in range(len(layers) - 1):

            #weights
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            #biases
            b = np.zeros(layers[i+1])
            #adding generated weight to object's properties
            self.weights.append(w)
            #adding generated bias to object's properties
            self.biases.append(b)

    def load_data(self, train_filename: str = "train_data_nn.csv", test_filename: str = "test_data_nn.csv") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data from a CSV file.
        """

        # get number of inputs
        input_size = self.layers[0]      
        # get number of outputs
        output_size = self.layers[-1]    

        #load in training data
        train_data = pd.read_csv(train_filename)
        #load in testing data
        test_data = pd.read_csv(test_filename)

        x_train = train_data.iloc[:, :input_size].values  
        y_train = train_data.iloc[:, -output_size:].values   
        x_test = test_data.iloc[:, :input_size].values
        y_test = test_data.iloc[:, -output_size:].values

        return x_train, y_train, x_test, y_test
    
    def _apply_activation(self, z:np.ndarray, activation: str) -> np.ndarray:
        """
        Apply the specified activation function.
        """

        #relu converts negative inputs to 0 and keeps positive inputs as they are
        if activation == "relu":
            return np.maximum(0, z)
        
        # sigmoid scales inputs between 0 and 1 for binary classification applications
        elif activation == "sigmoid":
            #to prevent overflow
            z = np.clip(z, -500, 500)
            return 1 / (1 + np.exp(-z))
        
        # tanh scales inputs between -1 and 1
        elif activation == "tanh":
            return np.tanh(z)
        
        # ReLU (leaky) which keeps positive inputs the same and scales the rest down by a factor of 0.01
        elif activation == "leaky_relu":
            return np.where(z > 0, z, z * 0.01)
        
        # softmax which converts inputs into a probability distribution
        elif activation == "softmax":
            if z.ndim == 1:
                z_shifted = z - np.max(z)
                exp_z = np.exp(z_shifted)
                return exp_z / np.sum(exp_z)
            else:
                z_shifted = z - np.max(z, axis=1, keepdims=True)
                exp_z = np.exp(z_shifted)
                return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x: np.ndarray) -> List[np.ndarray]:
        """
        Forward pass through the neural network.
        """

        #store all layer outputs and pre-activation values
        activations = [x]
        #before activation values for each layer
        z_values = []

        current_input = x

        #iterating through each layer
        for layer in range(len(self.weights)):
            
            #linear transformation
            z = current_input @ self.weights[layer] + self.biases[layer]
            z_values.append(z)

            #apply activation function
            activation_name = self.activations[layer]
            activated_output = self._apply_activation(z, activation_name)
            activations.append(activated_output)

            #step 3, output to input 
            current_input = activated_output

        return activations, z_values
    
    def _calculate_loss(self, y_actual: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the loss using Mean Squared Error (MSE).
        """

        #dep on last layer
        #if softmax (multiple outputs)
        if self.activations[-1] == "softmax":
            #cross-entropy loss
            y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
            return -np.mean(np.sum(y_actual * np.log(y_pred_clipped), axis=1))
    
        elif self.activations[-1] == "sigmoid":
            #binary cross entropy (2 ouptuts or binary classification)
            y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
            return -np.mean(y_actual * np.log(y_pred_clipped) + (1 - y_actual) * np.log(1 - y_pred_clipped))
        
        else:
            # for other types of output activation layers
            return np.mean((y_actual - y_pred) ** 2)
    
    def backward(self, x: np.ndarray, y: np.ndarray, activations: List[np.ndarray], z_values: List[np.ndarray]) -> None:
        """
        Backward pass - calculate gradients and update weights/biases.
        """
        m = x.shape[0] 
        
        #output layer error
        output_error = activations[-1] - y 

        # lists to store gradients
        weight_gradients = []
        bias_gradients = []
        
        # work backwards through layers
        current_error = output_error
        
        for layer in reversed(range(len(self.weights))):
            # gradient for current layer
            if layer == 0:
                # first hidden layer, use oriignal input
                dW = activations[layer].T @ current_error / m
            else:
                # iterate through for others
                dW = activations[layer].T @ current_error / m
                
            db = np.mean(current_error, axis=0)
            
            # Store gradients from beginning since we're going backwards
            weight_gradients.insert(0, dW)
            bias_gradients.insert(0, db)
            
            # Calculate error for previous layer (if not the first layer)
            if layer > 0:
                current_error = (current_error @ self.weights[layer].T) * \
                            self._activation_derivative(activations[layer], self.activations[layer-1])
        
        # Update weights and biases using gradients
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]

    def _activation_derivative(self, activated_values: np.ndarray, activation: str) -> np.ndarray:
        """
        Calculate derivative of activation function.
        """
        if activation == 'relu':
            return np.where(activated_values > 0, 1, 0)
        
        elif activation == 'sigmoid':
            # activated_values is already sigmoid output
            return activated_values * (1 - activated_values)
        
        elif activation == 'tanh':
            # activated_values is already tanh output
            return 1 - activated_values**2
        
        elif activation == 'leaky_relu':
            return np.where(activated_values > 0, 1, 0.01)
        
        else:
            # For linear/softmax
            return np.ones_like(activated_values)
        
    #all together
    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 100, verbose: bool = True) -> List[float]:
        """
        Train the neural network using the training set.
        """
        loss_history = []
        
        for epoch in range(epochs):
            # Forward pass
            activations, z_values = self.forward(X_train)
            
            # Calculate loss
            loss = self._calculate_loss(y_train, activations[-1])
            loss_history.append(loss)
            
            # Backward pass (update weights and biases)
            self.backward(X_train, y_train, activations, z_values)
            
            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.6f}")
        
        self.is_trained = True
        return loss_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
    
        """
        if not self.is_trained:
            print("Model not trained yet.")
    
        activations, _ = self.forward(X)
        return activations[-1]  

#I'm using an ai-generated main function to demonstrate

def main():
    """
    Main function to demonstrate neural network training and testing.
    """
    print("=" * 60)
    print("NEURAL NETWORK TRAINING AND TESTING")
    print("=" * 60)
    
    # Step 1: Initialize Neural Network
    print("\n🔧 STEP 1: Initializing Neural Network")
    print("-" * 40)
    layers = [4, 8, 6, 3]  # 4 inputs, 2 hidden layers, 3 outputs
    activations = ['relu', 'relu', 'softmax']
    learning_rate = 0.1
    
    nn = NeuralNetwork(layers, activations, learning_rate)
    print(f"✅ Network Architecture: {layers}")
    print(f"✅ Activations: {activations}")
    print(f"✅ learning Rate: {learning_rate}")
    print(f"✅ Total Parameters: {sum(w.size + b.size for w, b in zip(nn.weights, nn.biases))}")
    
    # Step 2: Load Data
    print("\n📊 STEP 2: Loading Training and Test Data")
    print("-" * 40)
    try:
        X_train, y_train, X_test, y_test = nn.load_data()
        print(f"✅ Training data loaded: {X_train.shape[0]} samples")
        print(f"✅ Test data loaded: {X_test.shape[0]} samples")
        print(f"✅ Input features: {X_train.shape[1]}")
        print(f"✅ Output classes: {y_train.shape[1]}")
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        print("📝 Make sure train_data_nn.csv and test_data_nn.csv exist in the current directory")
        return
    
    # Step 3: Test Forward Pass (Before Training)
    print("\n🔮 STEP 3: Testing Forward Pass (Before Training)")
    print("-" * 40)
    initial_predictions = nn.predict(X_test[:3])  # Test on first 3 samples
    print("✅ Sample predictions (untrained network):")
    for i, pred in enumerate(initial_predictions):
        print(f"   Sample {i+1}: [{pred[0]:.3f}, {pred[1]:.3f}, {pred[2]:.3f}]")
    
    # Step 4: Train the Network
    print("\n🎯 STEP 4: Training Neural Network")
    print("-" * 40)
    print("Starting training process...")
    
    epochs = 50
    loss_history = nn.train(X_train, y_train, epochs=epochs, verbose=True)
    
    print(f"\n✅ Training completed!")
    print(f"✅ Initial Loss: {loss_history[0]:.6f}")
    print(f"✅ Final Loss: {loss_history[-1]:.6f}")
    print(f"✅ Loss Reduction: {((loss_history[0] - loss_history[-1]) / loss_history[0] * 100):.2f}%")
    
    # Step 5: Test Predictions (After Training)
    print("\n🧪 STEP 5: Testing Predictions (After Training)")
    print("-" * 40)
    predictions = nn.predict(X_test)
    
    print("✅ Sample predictions (trained network):")
    for i, pred in enumerate(predictions):
        actual = y_test[i]
        predicted_class = np.argmax(pred)
        actual_class = np.argmax(actual)
        confidence = pred[predicted_class]
        
        print(f"   Sample {i+1}:")
        print(f"      Predicted: Class {predicted_class} (confidence: {confidence:.3f})")
        print(f"      Actual:    Class {actual_class}")
        print(f"      Correct:   {'✅ Yes' if predicted_class == actual_class else '❌ No'}")
    
    # Step 6: Calculate Accuracy
    print("\n📈 STEP 6: Final Performance Metrics")
    print("-" * 40)
    predicted_classes = np.argmax(predictions, axis=1)
    actual_classes = np.argmax(y_test, axis=1)
    accuracy = np.mean(predicted_classes == actual_classes)
    
    print(f"✅ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"✅ Correct Predictions: {np.sum(predicted_classes == actual_classes)}/{len(actual_classes)}")
    
    # Step 7: Summary
    print("\n🎉 STEP 7: Training Summary")
    print("-" * 40)
    print(f"✅ Network successfully trained with {epochs} epochs")
    print(f"✅ Loss decreased from {loss_history[0]:.6f} to {loss_history[-1]:.6f}")
    print(f"✅ Final test accuracy: {accuracy*100:.2f}%")
    print(f"✅ Model is ready for use!")
    
    print("\n" + "=" * 60)
    print("NEURAL NETWORK DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    main()

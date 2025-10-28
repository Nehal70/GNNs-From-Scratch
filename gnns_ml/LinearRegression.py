#imports
import numpy as np
import pandas as pd
from typing import Tuple, Union

class LinearRegression:
    def __init__(self):
        """
        Defining Slope, Intercept and whether the model is trained or not.
        """
        self.slope = None
        self.intercept = None
        self.is_trained = False

    def load_data(self, filename: str = "linear_regression_data.csv") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data from a CSV file.
        """
        data = pd.read_csv(filename)
        x_train = data.iloc[:, 0].values  # All columns except the last one
        y_train = data.iloc[:, 1].values   # Only the last column
        x_test = data.iloc[:, 2].values
        y_test = data.iloc[:, 3].values
        return x_train, y_train, x_test, y_test
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the linear regression model using the training dataset via method of least squares.
        """

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        x_sum = np.sum(x_train)
        y_sum = np.sum(y_train)
        xy_sum = np.sum(x_train * y_train)
        x2_sum = np.sum(x_train ** 2)
        length = len(x_train)

        #slope using mean squares formula
        slope = (length*xy_sum - x_sum*y_sum) / (length*x2_sum - x_sum**2)

        #intercept using mean squares formula
        intercept = (y_sum - slope*x_sum) / length

        #update object attributes
        self.slope = slope
        self.intercept = intercept
        self.is_trained = True

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the output for given input(s) using the trained model.
        """
        if not self.is_trained:
            raise Exception("Model is not trained yet. Please train the model before making predictions.")
        
        return self.slope * x + self.intercept

    def calculate_mse(self, y_actual: np.ndarray, y_pred:np.ndarray) -> float:
        """
        Calculate accuracy using Mean Squared Error (MSE).
        """
        return np.mean((y_actual - y_pred) ** 2)

    def calculate_r_squared(self, y_actual: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared (coefficient of determination)."""
        ss_res = np.sum((y_actual - y_pred) ** 2)  # Sum of squared residuals
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)  # Total sum of squares
        return 1 - (ss_res / ss_tot)

def main():
    """Main function to run the linear regression pipeline."""

    # Initialize the model
    model = LinearRegression()
    
    # Load data
    x_train, y_train, x_test, y_test = model.load_data()
    
    print(f"Training data shape: X={x_train.shape}, y={y_train.shape}")
    print(f"Testing data shape: X={x_test.shape}, y={y_test.shape}")
    
    # Train the model
    print("\nTraining the model")
    model.train(x_train, y_train)
    
    print(f"Slope: {model.slope:.4f}, Intercept: {model.intercept:.4f}")
    
    # Make predictions on test data
    print("\nPredictions:")
    y_pred = model.predict(x_test)
    
    return model

if __name__ == "__main__":
    model = main()      
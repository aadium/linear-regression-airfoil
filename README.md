# Linear Regression Breakdown

I have implemented an algorithm that uses gradient descent to train a linear regression model. I have implemented functions for calculating loss and gradient, calculating the model weights (a process which is called training), and evaluating the model's performance.

### Dataset Description

The [NASA 'airfoil' dataset](https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise#) was "obtained from a series of aerodynamic and acoustic tests of two and three-dimensional airfoil blade sections conducted in an anechoic [soundproof] wind tunnel." - R. Lopez, PhD

An airfoil is the cross-sectional shape of a wing, blade or propeller. For this particular dataset, specific variables were measured that are known to contribute to the amount of noise the airfoil generates when exposed to smooth air flow. For more information about the prediction of noise production from airfoils, check out this [link](https://ntrs.nasa.gov/citations/19890016302).

Inputs: 
- Frequency, in Hz
- Angle of attack, in degrees
- Chord length [length of airfoil], in meters
- Free-stream velocity, in meters per second
- Suction side displacement thickness, in meters

Output: 
- Scaled sound pressure level, in decibels.

Preview:

| Pressure (dB) | Frequency (Hz) | Angle (degrees) | Length  (m) | Velocity (m/s) | Displacement (m) 
|--|--|--|--|--|--|
| 126.201 | 800 | 0 | 0.3048 | 71.3 | 0.002663 
| 125.201 | 1000| 0 | 0.3048 | 71.3 | 0.002663 
| 125.951 | 1250| 0 | 0.3048 | 71.3 | 0.002663 

## API Documentation

### CoreFunctions Class

Static utility functions for linear regression mathematics:

#### Methods:
- `designMatrix(X)` - Adds bias term (column of ones) to feature matrix
- `initializeModelParameters(X)` - Initializes model parameters to zeros
- `loss(X, Y, theta)` - Calculates mean squared error loss
- `gradient(X, Y, theta)` - Computes gradient for gradient descent
- `update(theta, gradients, alpha)` - Updates parameters using gradient descent
- `train(X_train, Y_train, theta0, num_epochs, lr)` - Full training loop
- `predict(X, theta)` - Makes predictions using trained parameters

### LinearRegressionModel Class

Main gradient descent implementation:

#### Constructor:
```python
model = LinearRegressionModel()
```

#### Methods:
- `trainLRModel(X_train, Y_train, X_test, Y_test, dataset_name, num_epochs=1000, lr=0.01, verbose=True)`
  - Trains the model using gradient descent
  - Returns: `(theta, predictions, test_loss)`

### ClosedFormLinearRegressionModel Class

Analytical solution implementation:

#### Constructor:
```python
model = ClosedFormLinearRegressionModel()
```

#### Methods:
- `trainCLRModel(X_train, Y_train, X_test, Y_test, dataset_name, verbose=True)`
  - Solves using normal equations: θ = (X^T X)^(-1) X^T y
  - Returns: `(theta, predictions, test_loss)`

## Performance Metrics

The implementation provides comprehensive evaluation metrics:

- **Mean Absolute Error (MAE)**: Average absolute difference between predictions and actual values
- **Root Mean Square Error (RMSE)**: Square root of average squared differences
- **Total Loss**: Mean squared error loss value
- **Best/Worst Predictions**: Identifies highest and lowest error predictions
- **Sample-by-Sample Analysis**: Detailed comparison table for first/last 10 predictions

## Algorithm Implementation

### Gradient Descent
- **Loss Function**: Mean Squared Error (MSE)
- **Optimization**: Batch gradient descent
- **Learning Rate**: Configurable (default: 0.01)
- **Convergence**: Fixed number of epochs
- **Visualization**: Real-time loss plotting

### Closed-Form Solution
- **Method**: Normal equations
- **Formula**: θ = (X^T X)^(-1) X^T y
- **Advantages**: Exact solution, no hyperparameter tuning
- **Limitations**: Computationally expensive for large datasets, requires matrix inversion
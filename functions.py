import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(filename: str) -> tuple:
    df = pd.read_csv(filename)
    X = df.iloc[:, 1:].values
    Y = df.iloc[:, 0].values.reshape((-1, 1))
    return (X, Y)

def designMatrix(X: np.ndarray) -> np.ndarray:
    return np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)

def loss(X: np.ndarray, Y: np.ndarray, theta: np.ndarray) -> float:
    y_hat = X@theta
    errors = ((Y - y_hat)**2)/2
    return np.mean(errors)

def gradient(X: np.ndarray, Y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    indGradients = -((Y - X@theta) * X)
    return np.mean(indGradients, axis = 0).reshape(-1, 1)

def update(theta: np.ndarray, gradients: np.ndarray, alpha: float) -> np.ndarray:
    updatedTheta = theta - alpha * gradients # alpha is the learning rate
    return updatedTheta

def train(X_train: np.ndarray, Y_train: np.ndarray, theta0: np.ndarray, num_epochs: int, lr: float) -> np.ndarray:
    thetas = theta0.copy() # Thetas are the weights of the model that we will update during training
    losses = []  # To store loss values for each epoch
    
    for epoch in range(num_epochs):
        loss_value = loss(X_train, Y_train, thetas)
        losses.append(loss_value)  # Store the loss value for this epoch
        gradients = gradient(X_train, Y_train, thetas)
        thetas = update(thetas, gradients, lr)

        print(f"Epoch {epoch}, Loss: {loss_value:.4f}, Theta: {thetas.flatten()}")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs), losses, label='Training Loss', color='blue', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()  # This is crucial to display the plot
    
    return thetas

def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return X @ theta # Matrix multiplication to get predictions for each observation

def train_and_evaluate(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, 
                      dataset_name: str, num_epochs: int = 1000, lr: float = 0.01, verbose: bool = True) -> tuple:

    # Initial theta (weights)
    theta0 = np.zeros((X_train.shape[1], 1))
    if verbose:
        print(f"Initial Theta shape: {theta0.shape}, Initial Theta: {theta0.flatten()}")
    
    # Training the model
    if verbose:
        print(f"\nTraining {dataset_name} model...")
    theta = train(X_train, Y_train, theta0, num_epochs, lr)
    if verbose:
        print(f"Final Theta for {dataset_name}: {theta.flatten()}")
    
    # Testing the model
    predictions = predict(X_test, theta)
    test_loss = loss(X_test, Y_test, theta)
    
    if verbose:
        print(f"\n" + "="*50)
        print(f"          {dataset_name.upper()} RESULTS")
        print("="*50)
        
        # Create a comparison table
        actual = Y_test.flatten()
        predicted = predictions.flatten()
        errors = np.abs(actual - predicted)
        
        # Show first 10 and last 10 predictions to avoid too much output
        print(f"{'Index':<6} {'Actual':<10} {'Predicted':<12} {'Error':<10}")
        print("-" * 50)
        print("First 10 predictions:")
        for i in range(min(10, len(actual))):
            print(f"{i:<6} {actual[i]:<10.4f} {predicted[i]:<12.4f} {errors[i]:<10.4f}")
        
        if len(actual) > 10:
            print("...")
            print("Last 10 predictions:")
            for i in range(max(len(actual)-10, 10), len(actual)):
                print(f"{i:<6} {actual[i]:<10.4f} {predicted[i]:<12.4f} {errors[i]:<10.4f}")
        
        print("-" * 50)
        print(f"Mean Absolute Error: {np.mean(errors):.4f}")
        print(f"Root Mean Square Error: {np.sqrt(np.mean(errors**2)):.4f}")
        print(f"Total Loss: {test_loss:.4f}")
        
        # Summary statistics
        print(f"\nSUMMARY:")
        print(f"  Total samples: {len(actual)}")
        print(f"  Best Prediction (lowest error): Index {np.argmin(errors)}, Error: {np.min(errors):.4f}")
        print(f"  Worst Prediction (highest error): Index {np.argmax(errors)}, Error: {np.max(errors):.4f}")
        print(f"  Model Performance: {'Good' if np.mean(errors) < 0.5 else 'Needs Improvement'}")
        print("="*50)
    
    return theta, predictions, test_loss
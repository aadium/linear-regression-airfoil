import numpy as np

class CoreFunctions:
    @staticmethod
    def designMatrix(X: np.ndarray) -> np.ndarray:
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
    
    @staticmethod
    def initializeModelParameters(X: np.ndarray) -> np.ndarray:
        numParameters = X.shape[1]
        theta = np.zeros((numParameters, 1))
        return theta

    @staticmethod
    def loss(X: np.ndarray, Y: np.ndarray, theta: np.ndarray) -> float:
        y_hat = X@theta
        errors = ((Y - y_hat)**2)/2
        return np.mean(errors)

    @staticmethod
    def gradient(X: np.ndarray, Y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        indGradients = -((Y - X@theta) * X)
        return np.mean(indGradients, axis = 0).reshape(-1, 1)

    @staticmethod
    def update(theta: np.ndarray, gradients: np.ndarray, alpha: float) -> np.ndarray:
        updatedTheta = theta - alpha * gradients # alpha is the learning rate
        return updatedTheta

    @staticmethod
    def train(X_train: np.ndarray, Y_train: np.ndarray, theta0: np.ndarray, num_epochs: int, lr: float) -> np.ndarray:
        thetas = theta0.copy() # Thetas are the weights of the model that we will update during training
        losses = []  # To store loss values for each epoch
        
        for epoch in range(num_epochs):
            loss_value = CoreFunctions.loss(X_train, Y_train, thetas)
            losses.append(loss_value)  # Store the loss value for this epoch
            gradients = CoreFunctions.gradient(X_train, Y_train, thetas)
            thetas = CoreFunctions.update(thetas, gradients, lr)

            print(f"Epoch {epoch}, Loss: {loss_value:.4f}, Theta: {thetas.flatten()}")
        
        return thetas

    @staticmethod
    def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return X @ theta # Matrix multiplication to get predictions for each observation

class LinearRegressionModel:
    def __init__(self):
        self.core = CoreFunctions()
        self.theta = None
        self.is_trained = False

    def trainLRModel(self, X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, 
                        dataset_name: str, num_epochs: int = 1000, lr: float = 0.01, verbose: bool = True) -> tuple:

        # Design matrices for datasets
        X_train = self.core.designMatrix(X_train)
        X_test = self.core.designMatrix(X_test)
        
        # Initial theta (weights)
        theta0 = self.core.initializeModelParameters(X_train)
        if verbose:
            print(f"Initial Theta shape: {theta0.shape}, Initial Theta: {theta0.flatten()}")
        
        # Training the model
        if verbose:
            print(f"\nTraining {dataset_name} model...")
        theta = self.core.train(X_train, Y_train, theta0, num_epochs, lr)
        if verbose:
            print(f"Final Theta for {dataset_name}: {theta.flatten()}")
        
        # Store the trained parameters
        self.theta = theta
        self.is_trained = True
        
        # Testing the model
        predictions = self.core.predict(X_test, theta)
        test_loss = self.core.loss(X_test, Y_test, theta)
        
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
            print("="*50)
        
        return theta, predictions, test_loss

class ClosedFormLinearRegressionModel:
    def __init__(self):
        self.core = CoreFunctions()
        self.theta = None
        self.is_trained = False

    def trainCLRModel(self, X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, 
                        dataset_name: str, verbose: bool = True) -> tuple:
        X_design = self.core.designMatrix(X_train)
        X_test = self.core.designMatrix(X_test)
        theta = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ Y_train

        self.theta = theta
        self.is_trained = True
        if verbose:
            print(f"Final Theta for {dataset_name}: {theta.flatten()}")

        # Testing the model
        predictions = self.core.predict(X_test, theta)
        test_loss = self.core.loss(X_test, Y_test, theta)
        
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
            print("="*50)
        
        return theta, predictions, test_loss

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import sys 

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size=13, hidden_layers=[64, 32], output_size=3):
        """
        Initializes the MLP model.
        
        Parameters:
        - input_size: int, number of input features (default is 13).
        - hidden_layers: list, number of neurons in each hidden layer (default is [64, 32]).
        - output_size: int, number of output features (default is 3).
        """
        super(MLP, self).__init__()
        layers = []
        in_size = input_size
        # Create hidden layers based on the hidden_layers list
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_size, hidden_size))  # Add a linear layer
            layers.append(nn.ReLU())  # Add a ReLU activation function
            in_size = hidden_size
        # Output layer
        layers.append(nn.Linear(in_size, output_size))  # Final output layer
        self.model = nn.Sequential(*layers)  # Use nn.Sequential to create the complete model
        
    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Parameters:
        - x: torch.Tensor, input data.
        
        Returns:
        - torch.Tensor, output of the model.
        """
        return self.model(x)

    def train_model(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=50, learning_rate=0.001, save_dir="models/MLP_models/file_name"):
        """
        Trains the model using the provided training data.
        
        Parameters:
        - X_train: numpy array, training input data.
        - y_train: numpy array, training target data.
        - X_val: numpy array, validation input data.
        - y_val: numpy array, validation target data.
        - batch_size: int, number of samples per batch (default is 32).
        - epochs: int, number of epochs to train (default is 50).
        - learning_rate: float, learning rate for the optimizer (default is 0.001).
        - save_dir: str, directory to save the training loss curve and model (default is "models/MLP_models/file_name").
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Save hyperparameters to a file for tracking purposes
        hyperparameters_to_save = {
            'input_size': X_train.shape[1],
            'hidden_layers': [layer.out_features for layer in self.model if isinstance(layer, nn.Linear)][:-1],
            'output_size': y_train.shape[1],
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': learning_rate
        }
        hyperparameters_save_path = os.path.join(save_dir, "hyperparameters.txt")
        with open(hyperparameters_save_path, 'w') as f:
            f.write(str(hyperparameters_to_save))

        # Scale the data
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        X_train = self.scaler_X.fit_transform(X_train)  # Fit and transform input data
        y_train = self.scaler_y.fit_transform(y_train)  # Fit and transform target data
        X_val = self.scaler_X.transform(X_val)  # Transform validation input data
        y_val = self.scaler_y.transform(y_val)  # Transform validation target data

        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        # Create a dataset and data loader
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Define loss function and optimizer
        criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)  # Adam optimizer

        # Training loop with loss tracking
        self.train()  # Set the model to training mode
        train_loss_history = []
        val_loss_history = []
        for epoch in range(epochs):
            running_loss = 0.0
            for X_batch, y_batch in train_loader:
                # Zero the gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = self.forward(X_batch)
                # Compute loss
                loss = criterion(outputs, y_batch)
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_train_loss = running_loss / len(train_loader)
            train_loss_history.append(avg_train_loss)

            # Validation pass
            self.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                val_outputs = self.forward(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
                val_loss_history.append(val_loss)

            # Print average loss for the epoch
            sys.stdout.write("\rEpoch [{}/{}], Training Loss: {}, Validation Loss: {}".format(epoch+1, epochs, round(avg_train_loss,4), round(val_loss,4)))
            # Plot and save the loss curves
            plt.figure()
            plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Training Loss')
            plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            # plt.title('Training and Validation Loss Curve')
            plt.legend()
            plt.grid(True)
            loss_plot_path = os.path.join(save_dir, "metric.png")
            plt.savefig(loss_plot_path)
            # print(f"Training and validation loss curve saved at: {loss_plot_path}")
            plt.close()

    def evaluate_model(self, X_test, y_test):
        """
        Evaluates the model using the provided test data.
        
        Parameters:
        - X_test: numpy array, test input data.
        - y_test: numpy array, test target data.
        """
        # Scale the test data using the same scaler used for training data
        X_test = self.scaler_X.transform(X_test)  # Transform input data
        y_test = self.scaler_y.transform(y_test)  # Transform target data

        # Convert numpy arrays to PyTorch tensors
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(self.device)

        # Evaluate the model
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            predictions = self.forward(X_test_tensor)  # Get model predictions
            # Inverse transform the predictions and targets to original scale
            predictions = self.scaler_y.inverse_transform(predictions.cpu().numpy())
            y_test_original = self.scaler_y.inverse_transform(y_test_tensor.cpu().numpy())
            # Calculate Mean Squared Error for evaluation
            mse = np.mean((predictions - y_test_original) ** 2)
            print(f"Test MSE: {mse:.4f}")

    def save_model(self, save_dir="models/MLP_models/file_name"):
        """
        Saves the trained model and scalers to the specified directory.
        
        Parameters:
        - save_dir: str, directory to save the model and scalers (default is "models/MLP_models/file_name").
        """
        os.makedirs(save_dir, exist_ok=True)
        # Save the model
        model_path = os.path.join(save_dir, "mlp_model.pth")
        torch.save(self.state_dict(), model_path)  # Save model parameters
        # Save the scalers
        scaler_X_path = os.path.join(save_dir, "scaler_X.pkl")
        scaler_y_path = os.path.join(save_dir, "scaler_y.pkl")
        joblib.dump(self.scaler_X, scaler_X_path)  # Save input data scaler
        joblib.dump(self.scaler_y, scaler_y_path)  # Save target data scaler
        print("\nModel and scalers saved in folder: {}".format(save_dir))

    def set_device(self, device):
        """
        Sets the device to either GPU or CPU as specified by the user.
        
        Parameters:
        device (str): The desired device, either 'cuda' for GPU or 'cpu' for CPU.
        """
        if device not in ['cpu', 'cuda:0', 'cuda:1', 'cuda:2']:
            raise ValueError("Invalid device specified. Use 'cpu' or 'cuda'.")
        self.device = torch.device(device)
        self.to(self.device)  # Move model to the selected device

############## Functions 

def load_and_predict(X_test, model_dir, device='cpu'):
    """
    Loads the trained MLP model and makes predictions on the given test data.
    
    Parameters:
    - X_test: numpy array, input test data.
    - model_dir: str, directory where the trained model and scalers are saved (default is "models/MLP_models/file_name").
    - device: str, the device to use ('cpu' or 'cuda').
    
    Returns:
    - numpy array, predictions from the model (in original scale).
    """
    # Load the saved scalers
    scaler_X_path = f"{model_dir}/scaler_X.pkl"
    scaler_y_path = f"{model_dir}/scaler_y.pkl"
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

    # Load the hyperparameters
    hyperparameters_path = f"{model_dir}/hyperparameters.txt"
    if os.path.exists(hyperparameters_path):
        with open(hyperparameters_path, 'r') as f:
            hyperparameters = eval(f.read())  # Assuming the text file has a dictionary-like structure
            try:
                input_size = hyperparameters['input_size']
                hidden_layers = hyperparameters['hidden_layers']
                output_size = hyperparameters['output_size']
            except KeyError as e:
                raise ValueError(f"Missing hyperparameter: {str(e)} in the hyperparameters file.")
    else:
        raise ValueError("Hyperparameters file not found. Cannot load model configuration.")

    # Load the model
    model = MLP(input_size=input_size, hidden_layers=hidden_layers, output_size=output_size)  # Use parameters from file
    model_path = f"{model_dir}/mlp_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)

    # Scale the test data using the loaded scaler
    X_test_scaled = scaler_X.transform(X_test)

    # Convert numpy arrays to PyTorch tensors
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    # Get predictions
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predictions = model.forward(X_test_tensor)
        # Inverse transform the predictions to original scale
        predictions = scaler_y.inverse_transform(predictions.cpu().numpy())

    return predictions

import MLP.MLP_module as mlp
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import torch.nn as nn
import json
import torch

def parse_arguments():
    parser = argparse.ArgumentParser(description="Arguments for training a neural network model.")
    
    # Adding arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help="Path to the dataset to be used.")
    parser.add_argument('--model_save_folder_path', type=str, required=True,
                        help="Path where the model will be saved.")
    parser.add_argument('--num_neurons', type=str, required=True,
                        help="Comma-separated list of neurons per layer, or a single value for all layers.")
    parser.add_argument('--num_layers', type=int, required=False,
                        help="Number of layers in the neural network. Ignored if a list of neurons is provided.")
    parser.add_argument('--device', type=str, default=None, required = False,
                        help="Device to be used for training ('cpu' or 'cuda').")
    parser.add_argument('--data_split', type=float, default=0.2,
                        help="Fraction of the dataset to be used as validation data.")
    parser.add_argument('--random_state', type=int, default=None,
                        help="Random state for reproducibility of data splitting.")
    parser.add_argument('--epochs', type=int, default=50,
                        help="Number of epochs for training.")
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help="Learning rate for the optimizer.")
    parser.add_argument('--batch_size', type=int, default=50000,
                        help="Batch size for training.")
    parser.add_argument('--save_model_during_training', action='store_true',
                        help="Flag to save the model at intervals during training.")
    parser.add_argument('--save_step', type=int, default=100,
                        help="Frequency (in epochs) at which the model is saved during training.")
    parser.add_argument('--scheduler_type', type=str, default='StepLR',
                        help='Scheduler type: StepLR, ExponentialLR, CosineAnnealingLR')
    parser.add_argument('--scheduler_params', type=str, default='{}',
                        help='Scheduler parameters as a JSON string (e.g. \'{"step_size": 10, "gamma": 0.5}\')')
    parser.add_argument('--activation_fn', type=str, default='ReLU',
                        help='Activation function: ReLU, Sigmoid, Tanh')
    
    # Parse the arguments
    args = parser.parse_args()

    # Determine which device to use
    if args.device is None:
        # No user-specified device, use hierarchical strategy
        preferred_devices = [0, 1, 2]
        device = "cpu"  # Default to CPU
    
        if torch.cuda.is_available():
            # Find the highest-priority available device
            for idx in preferred_devices:
                if idx < torch.cuda.device_count():
                    device = f"cuda:{idx}"
                    break
    else:
        # User specified a device
        device = args.device
    args.device = device
    print(f"Using device: {device}")
    # Set activation function based on argument
    activation_fn_dict = {
        'ReLU': nn.ReLU,
        'Sigmoid': nn.Sigmoid,
        'Tanh': nn.Tanh
    }
    if args.activation_fn not in activation_fn_dict:
        raise ValueError(f"Unsupported activation function: {args.activation_fn}. Supported functions: {list(activation_fn_dict.keys())}")
    
    return args

def train_mlp(data_path, model_save_folder, neurons, activation_fn = nn.ReLU, layers=None, 
              device='cpu', data_split=0.2, random_state=42,
              epochs=50, learning_rate=0.001, batch_size=50000, 
              save_model_during_training=False, save_step=100, 
              scheduler_type='StepLR', scheduler_params=None):
    # Load the dataset
    GW_data = pd.read_csv(data_path)
    GW_data['geocent_time'] = GW_data['geocent_time'] % 86164.1

    # Split the DataFrame into input features (X) and target values (y)
    X = GW_data[['mass_1', 'mass_2', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'ra', 'dec', 'theta_jn', 'phi_jl', 'phi_12',
                  'psi', 'geocent_time']].values
    y = (GW_data[['snr_H1', 'snr_L1', 'snr_V1']].values * GW_data[['luminosity_distance']].values).squeeze()

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=data_split, random_state=random_state)

    # Process num_neurons argument
    try:
        neuron_list = [int(x) for x in neurons.split(',')]
        neurons = neuron_list
        layers = len(neuron_list)
    except ValueError:
        if layers is None:
            raise ValueError("Please specify the number of layers if neurons is not a list.")
        neurons = [int(neurons)] * layers

    # Initialize and set device for the model
    model = mlp.MLP(input_size=13, hidden_layers=neurons, output_size=3, activation_fn = activation_fn)
    model.set_device(device=device)

    # Train the model with training and validation data
    model.train_model(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                      epochs=epochs, learning_rate=learning_rate,
                      batch_size=batch_size, save_dir="models/MLP_models/" + model_save_folder,
                      save_model_during_training=save_model_during_training, save_step=save_step, scheduler_type=scheduler_type, scheduler_params=scheduler_params)

if __name__ == "__main__":
    args = parse_arguments()
    # # Convert scheduler_params from JSON string to dictionary
    # scheduler_params = json.loads(args.scheduler_params)
    # Parse the scheduler parameters string into a dictionary
    scheduler_params_str = args.scheduler_params
    scheduler_params = {}

    if scheduler_params_str:
        try:
            # Split the string by spaces to get key-value pairs
            key_value_pairs = scheduler_params_str.split(',')
            for pair in key_value_pairs:
                key, value = pair.split('=')
                # Convert the value to float or int if possible
                if '.' in value or 'e' in value:
                    value = float(value)
                else:
                    value = int(value)
                scheduler_params[key] = value
        except ValueError as e:
            print(f"Error parsing scheduler parameters: {e}")
            
    # Print the arguments to verify
    print("Data Path:", args.data_path)
    print("Model Save Folder Path:", args.model_save_folder_path)
    print("Number of Neurons per Layer:", args.num_neurons)
    print("Number of Layers:", args.num_layers)
    print("Device:", args.device)
    print("Data Split Ratio:", args.data_split)
    print("Random State:", args.random_state)
    print("Number of Epochs:", args.epochs)
    print("Learning Rate:", args.learning_rate)
    print("Batch Size:", args.batch_size)
    print("Save Model During Training:", args.save_model_during_training)
    print("Save Step:", args.save_step)
    print("Scheduler Type:", args.scheduler_type)
    print("Scheduler Parameters:", scheduler_params)

    
    # Train the MLP model
    train_mlp(data_path=args.data_path,
              model_save_folder=args.model_save_folder_path,
              neurons=args.num_neurons,
              layers=args.num_layers,
              device=args.device,
              data_split=args.data_split,
              random_state=args.random_state,
              epochs=args.epochs,
              learning_rate=args.learning_rate,
              batch_size=args.batch_size,
              save_model_during_training=args.save_model_during_training,
              save_step=args.save_step,
             scheduler_type=args.scheduler_type, 
             scheduler_params=scheduler_params) 

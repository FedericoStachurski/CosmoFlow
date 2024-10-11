import os
import sys
import torch
import pandas as pd
import numpy as np
import copy
import pickle
import shutil
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from glasflow.flows import RealNVP, CouplingNSF
import argparse
from tqdm import tqdm
from scipy.stats import norm
# Get the current script directory
current_dir = os.path.dirname(os.path.realpath(__file__))

# Get the parent directory (where cosmology_functions is located)
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)
from cosmology_functions import utilities, cosmology


class DataLoaderClass:
    def __init__(self, data_path, batches_data=1, xyz=True, scaler_type='MinMax', n_conditional=1):
        # Initialize the DataLoaderClass with the number of data batches, coordinate type, scaler type, and number of conditional variables.
        self.data_path = '../data_cosmoflow/'+data_path+'.csv'
        self.batches_data = batches_data
        self.xyz = xyz
        self.scaler_type = scaler_type
        self.n_conditional = n_conditional
        self.data = None
        self.scaler_x = None
        self.scaler_y = None

    def load_data(self):
        print("Loading data batches...")
        # Read and concatenate data from multiple batches
        list_data = []
        for i in range(self.batches_data):
            print(f"Reading batch {i + 1}")
            list_data.append(self._read_data(i + 1))
        
        # Concatenate and shuffle data to ensure randomness
        GW_data = pd.concat(list_data).drop_duplicates(keep='first').sample(frac=1)
        print("Data loaded and concatenated successfully.")
        self.data = self._process_data(GW_data)

    def _read_data(self, batch_number):

        # Get the current working directory
        current_directory = os.getcwd()
        
        print("Current Directory:", current_directory)
        # Load data from a specific batch number
        data_name = self.data_path.format(batch_number)
        print(f"Loading data from {data_name}")
        GW_data = pd.read_csv(data_name, skipinitialspace=True)
        return GW_data

    def _process_data(self, GW_data):
        print("Processing data...")
        # Adjust data columns and convert coordinates if needed
        GW_data['geocent_time'] = utilities.convert_gps_sday(GW_data['geocent_time'])
        
        if self.xyz:
            # Convert to Cartesian coordinates if requested
            print("Converting to Cartesian coordinates...")
            dl = np.array(GW_data['luminosity_distance'])
            ra = np.array(GW_data['ra'])
            dec = np.array(GW_data['dec'])
            x, y, z = cosmology.spherical_to_cart(dl, ra, dec)
            GW_data.loc[:, 'xcoord'] = x
            GW_data.loc[:, 'ycoord'] = y
            GW_data.loc[:, 'zcoord'] = z
            # Keep only the necessary columns for training
            GW_data = GW_data[['xcoord', 'ycoord', 'zcoord', 'mass_1', 'mass_2', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'theta_jn', 'phi_jl', 'phi_12', 'psi', 'geocent_time', 'H0']]
        else:
            # Keep polar coordinates if Cartesian conversion is not needed
            GW_data = GW_data[['luminosity_distance', 'ra', 'dec', 'mass_1', 'mass_2', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'theta_jn', 'phi_jl', 'phi_12', 'psi', 'geocent_time', 'H0']]
        print("Data processing complete.")
        return GW_data

    def scale_data(self):
        print("Scaling data...")
        # Scale data using the specified scaler (MinMax or Standard)
        self.scaler_x, self.scaler_y, scaled_data = utilities.scale_data(self.data, self.scaler_type, self.n_conditional)
        print("Data scaling complete.")
        return scaled_data

class TrainFlowClass:
    def __init__(self, args, data_loader):
        # Initialize the training class with parsed arguments and a data loader instance
        self.args = args
        self.data_loader = data_loader
        self.device = torch.device(args['device'])
        self.batch_size = int(args['batch_size'])
        self.flow_type = args['flow_type']
        self.epochs = int(args['epochs'])
        self.lr = float(args['learning_rate'])
        self.lr_scheduler = args['learning_rate_scheduler']
        self.dp = float(args['drop_out'])
        self.save_steps = int(args['save_step'])
        self.vp = args['Volume_preserving']
        self.linear_transform = args.get('linear_transform', None)
        self.n_conditional = int(args['n_conditional'])
        self.n_neurons = int(args['neurons'])
        self.n_transforms = int(args['layers'])
        self.n_blocks_per_transform = int(args['nblock'])
        self.scaler_x, self.scaler_y = data_loader.scaler_x, data_loader.scaler_y
        self.flow = None
        self.folder_name = args['Name_folder']
        self.path = 'trained_flows_and_curves/'
        self.train_size = float(args['train_size'])

        # Create the folder for saving the model if it doesn't exist
        if os.path.exists(self.path + self.folder_name):
            shutil.rmtree(self.path + self.folder_name)
        os.mkdir(self.path + self.folder_name)
        os.mkdir(self.path + self.folder_name + '/flows_epochs')

        # Save model hyperparameters
        para = {'batch_size': self.batch_size,
                'n_epochs': self.epochs,
                'shuffle': True,
                'activation': None,
                'dropout': self.dp,
                'learning_rate': self.lr,
                'optimizer': 'Adam',
                'linear_transform': self.linear_transform,
                'n_neurons': self.n_neurons,
                'n_transforms': self.n_transforms,
                'n_blocks_per_transform': self.n_blocks_per_transform,
                'n_inputs': None,  # Placeholder, will be updated after initialization
                'n_conditional_inputs': self.n_conditional,
                'flow_type': self.flow_type,
                'log_it': args['log_it'],
                'xyz': args['xyz'],
                'scaler': args['Scaler'],
                'lr_scheduler': self.lr_scheduler,
                'volume_preserving': self.vp
                }

        with open(self.path + self.folder_name + '/hyperparameters.txt', 'w') as f:
            f.write(str(para))

    def prepare_data(self):
        print("Preparing data for training and validation...")
        # Split the data into training and validation sets
        scaled_data = self.data_loader.scale_data()
        x_train, x_val = train_test_split(scaled_data, test_size=(1.0 - self.train_size), train_size=self.train_size)


        # Create PyTorch datasets and dataloaders for training and validation
        train_tensor = torch.from_numpy(np.asarray(x_train).astype('float32'))
        val_tensor = torch.from_numpy(np.asarray(x_val).astype('float32'))
        
        X_scale_train = train_tensor[:, :-self.n_conditional]  # Features for training
        Y_scale_train = train_tensor[:, -self.n_conditional:]  # Conditionals for training
        
        X_scale_val = val_tensor[:, :-self.n_conditional]  # Features for validation
        Y_scale_val = val_tensor[:, -self.n_conditional:]  # Conditionals for validation

        # Wrap the datasets in DataLoader for batch processing
        train_dataset = TensorDataset(X_scale_train, Y_scale_train)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        val_dataset = TensorDataset(X_scale_val, Y_scale_val)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Store validation data for evaluation
        self.X_scale_val = X_scale_val
        self.Y_scale_val = Y_scale_val
        print("Data preparation complete.")

    def initialize_flow(self):
        print("Initializing flow model...")
        # Initialize the flow model based on the type specified
        n_inputs = len(self.data_loader.data.columns) - self.n_conditional
        if self.flow_type == 'RealNVP':
            # Initialize RealNVP model with the given parameters
            self.flow = RealNVP(
                n_inputs=n_inputs,
                n_transforms=self.n_transforms,
                n_neurons=self.n_neurons,
                n_conditional_inputs=self.n_conditional,
                n_blocks_per_transform=self.n_blocks_per_transform,
                batch_norm_between_transforms=True,
                dropout_probability=self.dp,
                linear_transform=self.linear_transform,
                volume_preserving=self.vp
            )
        elif self.flow_type == 'CouplingNSF':
            # Initialize CouplingNSF model with the given parameters
            self.flow = CouplingNSF(
                n_inputs=n_inputs,
                n_transforms=self.n_transforms,
                n_neurons=self.n_neurons,
                n_conditional_inputs=self.n_conditional,
                n_blocks_per_transform=self.n_blocks_per_transform,
                batch_norm_between_transforms=True,
                dropout_probability=self.dp,
                linear_transform=self.linear_transform
            )
        else:
            raise ValueError('Flow not implemented')

        # Update the saved hyperparameters with actual number of inputs
        with open(self.path + self.folder_name + '/hyperparameters.txt', 'a') as f:
            f.write(f"\nn_inputs: {n_inputs}")
        print("Flow model initialized.")

    def _validate(self, loss_dict, kl_dict):
        self.flow.eval()  # Set model to evaluation mode
        val_loss = 0
        with torch.no_grad():  # No need to compute gradients during validation
            for batch_idx, batch in enumerate(self.val_loader):
                # print(f"Validating batch {batch_idx + 1}/{len(self.val_loader)}")
                target_val, conditionals_val = batch
                loss = -self.flow.log_prob(target_val.to(self.device), conditional=conditionals_val.to(self.device)).mean()
                val_loss += loss.item()
                # print(f"Validation Batch {batch_idx + 1} loss: {loss.item():.4f}")
        val_loss /= len(self.val_loader)  # Average validation loss for the epoch
        loss_dict['val'].append(val_loss)
    
        # Calculate KL divergence for latent space evaluation
        with torch.no_grad():
            conditionals = self.Y_scale_val.to(self.device)
            target_data = self.X_scale_val.to(self.device)
            latent_samples, _ = self.flow.forward(target_data[:, :len(target_data[0])], conditional=conditionals)
            z_ = latent_samples.cpu().detach().numpy()[:10000]  # Use 10,000 samples from latent space for KL evaluation
    
            # Evaluate KL divergence between each latent dimension and a Gaussian
            g = np.linspace(-5, 5, 1000)
            gaussian = norm.pdf(g)
            for i in range(len(z_[0])):
                _, kl_vals = utilities.KL_evaluate_gaussian(z_[:, i], gaussian, g)
                kl_dict[f'KL_vals{i+1}'].append(kl_vals)
            latent_samples = dict(z_samples=z_)
        return val_loss, kl_dict, latent_samples


    def train(self):
        print("Starting training process...")
        self.flow.to(self.device)  # Move the model to the specified device (CPU/GPU)
        optimiser = torch.optim.Adam(self.flow.parameters(), lr=self.lr, weight_decay=0)  # Initialize the optimizer
        best_model = copy.deepcopy(self.flow.state_dict())  # Keep a copy of the best model
        best_val_loss = np.inf  # Set initial validation loss to infinity

        loss_dict = {'train': [], 'val': []}  # Dictionary to store training and validation loss
        kl_dict = {f'KL_vals{i+1}': [] for i in range(len(self.data_loader.data.columns) - self.n_conditional)}  # Dictionary to store KL divergence values

        for epoch in range(self.epochs):
            self.flow.train()  # Set model to training mode
            train_loss = 0
            for batch_idx, batch in enumerate(self.train_loader):
                target_train, conditionals_train = batch
                optimiser.zero_grad()  # Zero gradients
                loss = -self.flow.log_prob(target_train.to(self.device), conditional=conditionals_train.to(self.device)).mean()  
                # Compute negative log-likelihood loss
                loss.backward()  # Backpropagate loss
                optimiser.step()  # Update weights
                train_loss += loss.item()

            train_loss /= len(self.train_loader)  # Average training loss for the epoch
            loss_dict['train'].append(train_loss)  # Store training loss
            val_loss, kl_dict, latent_samples = self._validate(loss_dict, kl_dict)  # Perform validation after each epoch
            # Extract the last value of each list
            last_entries = [values[-1] for values in kl_dict.values()]
            
            # Calculate the mean of the last entries
            mean_last_entries = sum(last_entries) / len(last_entries)

            sys.stdout.write(f"\rEpoch {epoch + 1}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Mean KL: {mean_last_entries:.4f}")
            sys.stdout.flush()


            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_model = copy.deepcopy(self.flow.state_dict())
                best_val_loss = val_loss

            # Save model checkpoint periodically
            if (epoch + 1) % self.save_steps == 0:
                model_path = f'{self.path + self.folder_name}/flows_epochs/flow_epoch_{epoch + 1}.pt'
                torch.save(self.flow.state_dict(), model_path)


            #save loss and kl values data in pickle files 
            with open(self.path+self.folder_name+'/loss_data.pickle', 'wb') as handle:
                pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.path+self.folder_name+'/kl_data.pickle', 'wb') as handle:
                pickle.dump(kl_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.path+self.folder_name+'/latent_data.pickle', 'wb') as handle:
                pickle.dump(latent_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)



# Main function to run the entire pipeline
if __name__ == "__main__":
    # Argument parsing
    ap = argparse.ArgumentParser()
    ap.add_argument("-data_path", "--data_path", required=True,
                    help="Path to the CSV data file, with batch placeholder if multiple batches are used.")
    ap.add_argument("-Name", "--Name_folder", required=True,
                    help="Name of the folder to save the FLOW model")
    ap.add_argument("-batch", "--batch_size", required=False, default=5000,
                    help="Batch size of the data to pass")
    ap.add_argument("-n_cond", "--n_conditional", required=False, default=1,
                    help="How many conditional variables")
    ap.add_argument("-train_size", "--train_size", required=False, default=0.75,
                    help="Percentage of training data")
    ap.add_argument("-flow_type", "--flow_type", required=False, default="RealNVP",
                    help="Type of flow model to use, e.g., RealNVP or CouplingNSF")
    ap.add_argument("-epochs", "--epochs", required=False, default=1000,
                    help="Number of training iterations")
    ap.add_argument("-neurons", "--neurons", required=False, default=128,
                    help="Number of neurons in layer")
    ap.add_argument("-layers", "--layers", required=False, default=6,
                    help="Number of hidden layers")
    ap.add_argument("-nblock", "--nblock", required=False, default=4,
                    help="Number of blocks per layer")
    ap.add_argument("-lr", "--learning_rate", required=False, default=0.001,
                    help="Learning rate for training")
    ap.add_argument("-lr_scheduler", "--learning_rate_scheduler", required=False, default='No',
                    help="Type of learning rate scheduler to use")
    ap.add_argument("-linear_transform", "--linear_transform", required=False, default=None,
                    help="Type of linear transformation to use, if any")
    ap.add_argument("-drop_out", "--drop_out", required=False, default=0.0,
                    help="Dropout rate")
    ap.add_argument("-log_it", "--log_it", required=False, default=0,
                    help="Whether to apply logit transformation to uniform distributions")
    ap.add_argument("-device", "--device", required=True,
                    help="Which CUDA device to use (e.g., cuda:0, cuda:1)")
    ap.add_argument("-xyz", "--xyz", required=True, default=1,
                    help="Convert to xyz coordinates: 1 if yes, else 0")
    ap.add_argument("-Scaler", "--Scaler", required=True, default='MinMax',
                    help="Type of scaler to use: MinMax or Standard")
    ap.add_argument("-save_step", "--save_step", required=False, default=100,
                    help="How often (in epochs) to save the model")
    ap.add_argument("-batches_data", "--batches_data", required=False, default=1,
                    help="Number of data batches to use")
    ap.add_argument("-VP", "--Volume_preserving", required=True, default=False,
                    help="Whether to enable volume preserving flow")
    args = vars(ap.parse_args())

    # Data loading and preprocessing
    print("Initializing DataLoaderClass...")
    data_loader = DataLoaderClass(data_path=args["data_path"],
                                  batches_data=int(args["batches_data"]),
                                  xyz=bool(int(args["xyz"])),
                                  scaler_type=args["Scaler"],
                                  n_conditional=int(args["n_conditional"]))

    # Load and process data
    data_loader.load_data()

    # Flow training
    print("Initializing TrainFlowClass...")
    train_flow = TrainFlowClass(args, data_loader)

    # Prepare data for training
    train_flow.prepare_data()

    # Initialize the flow model
    train_flow.initialize_flow()

    # Start training the flow model
    train_flow.train()

    print("Training complete.")

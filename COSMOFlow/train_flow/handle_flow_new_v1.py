import os
import pickle
import torch
import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
# Get the current script directory
current_dir = os.path.dirname(os.path.realpath(__file__))

# Get the parent directory (where cosmology_functions is located)
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from glasflow.flows import RealNVP, CouplingNSF
from cosmology_functions import utilities, cosmology
from scipy.special import logsumexp


class HandleFlow:
    def __init__(self, path, flow_name, device, epoch=None, threads=1, conditional=True):
        """
        Initialize the HandleFlow class.

        Parameters:
            path (str): Path to the model directory.
            flow_name (str): Name of the flow model.
            device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
            epoch (int, optional): Specific epoch to load. Defaults to None for the latest.
            threads (int, optional): Number of threads for PyTorch. Defaults to 1.
            conditional (bool, optional): Whether the flow is conditional. Defaults to True.
        """
        self.path = path
        self.flow_name = flow_name
        self.device = device
        self.epoch = epoch
        self.threads = threads
        self.conditional = conditional
        torch.set_num_threads(self.threads)
        self.flow, self.hyperparameters, self.scaler_x, self.scaler_y = self.load_hyperparameters_scalers_flow()
        self.logit = int(self.hyperparameters['log_it'])
    
    def load_hyperparameters_scalers_flow(self):
        torch.set_num_threads(self.threads)
        # Open hyperparameter dictionary
        path = self.path
        flow_name = self.flow_name
        hyper = open(path + flow_name + '/' + 'hyperparameters.txt', 'r').read()
        hyperparameters = eval(hyper)

        device = 'cpu'
        n_inputs = hyperparameters['n_inputs']
        n_conditional_inputs = hyperparameters['n_conditional_inputs']
        n_neurons = hyperparameters['n_neurons']
        n_transforms = hyperparameters['n_transforms']
        n_blocks_per_transform = hyperparameters['n_blocks_per_transform']
        dropout = hyperparameters['dropout']
        flow_type = hyperparameters['flow_type']

        # Open scaler_x and scaler_y
        scalerfile_x = path + flow_name + '/' + 'scaler_x.sav'
        scaler_x = pickle.load(open(scalerfile_x, 'rb'))

        if n_conditional_inputs != 0:
            scalerfile_y = path + flow_name + '/' + 'scaler_y.sav'
            scaler_y = pickle.load(open(scalerfile_y, 'rb'))
        else:
            scaler_y = None

        # Open flow model file flow.pt
        if self.epoch is None:
            flow_load = torch.load(path + flow_name + '/' + 'flow.pt', map_location=self.device)
        else:
            flow_load = torch.load(path + flow_name + '/flows_epochs/' + 'flow_epoch_{}.pt'.format(self.epoch), map_location=self.device)

        # Initialize the appropriate flow model
        if flow_type == 'RealNVP':
            flow_empty = RealNVP(
                n_inputs=n_inputs,
                n_transforms=n_transforms,
                n_neurons=n_neurons,
                n_conditional_inputs=n_conditional_inputs,
                n_blocks_per_transform=n_blocks_per_transform,
                batch_norm_between_transforms=True,
                dropout_probability=dropout,
                linear_transform='lu'
            )
        elif flow_type == 'CouplingNSF':
            flow_empty = CouplingNSF(
                n_inputs=n_inputs,
                n_transforms=n_transforms,
                n_neurons=n_neurons,
                n_conditional_inputs=n_conditional_inputs,
                n_blocks_per_transform=n_blocks_per_transform,
                batch_norm_between_transforms=True,
                dropout_probability=dropout,
                linear_transform='lu'
            )
        else:
            raise ValueError("Unsupported flow type")

        flow_empty.load_state_dict(flow_load)
        flow = flow_empty
        flow.eval()

        if n_conditional_inputs != 0:
            return flow, hyperparameters, scaler_x, scaler_y
        else:
            return flow, hyperparameters, scaler_x

    def sample_flow(self, conditional, n_samples):
        """
        Sample from the flow given the conditional input.

        Parameters:
            conditional (list or array): Conditional input values.
            n_samples (int): Number of samples to draw.

        Returns:
            np.ndarray: Samples from the flow.
        """
        conditional = np.array(conditional).T
        conditional_scaled = self.scaler_y.transform(conditional.reshape(-1, self.hyperparameters['n_conditional_inputs']))
        data_scaled = torch.from_numpy(conditional_scaled.astype('float32'))
        self.flow.to('cpu')
        with torch.no_grad():
            samples = self.flow.sample(n_samples, conditional=data_scaled)
            if self.logit == 1:
                samples = utilities.inverse_logit_transform(samples.numpy())
                samples = self.scaler_x.inverse_transform(samples)
            else:
                samples = self.scaler_x.inverse_transform(samples.numpy())
            
        return samples

    def evaluate_log_prob(self, target, conditional=None):
        """
        Evaluate the log-probability of the target given the conditional.

        Parameters:
            target (np.ndarray or pd.Series): Target values for evaluation.
            conditional (np.ndarray, optional): Conditional values. Defaults to None.

        Returns:
            np.ndarray: Log-probability values.
        """
        # Ensure target is a numpy array
        if isinstance(target, pd.Series):
            target = target.to_numpy()
        
        # Data scaled first, then logit if requested
        # Scale target data appropriately
        #### Data should be logit first then scaled!!!
        target_scaled = self.scaler_x.transform(target.reshape(-1, self.hyperparameters['n_inputs']) if target.ndim == 1 else target)
        if self.logit == 1:
            target_scaled = utilities.logit_transform(target_scaled)
        
        target_tensor = torch.from_numpy(target_scaled.astype('float32')).float().to(self.device)
        self.flow.to(self.device)
        with torch.no_grad():
            if conditional is not None:
                if isinstance(conditional, pd.Series) or np.isscalar(conditional):
                    conditional = np.array(conditional).reshape(-1, 1)
                elif conditional.ndim == 1:
                    conditional = conditional.reshape(-1, 1)

                # Use the reshape_conditional function to repeat the conditional if necessary
                if target_tensor.shape[0] > 1 and conditional.shape[0] == 1:
                    conditional = self.reshape_conditional(conditional, target_tensor.shape[0], axis=0)
                    
                conditional_scaled = self.scaler_y.transform(conditional)
                conditional_tensor = torch.from_numpy(conditional_scaled.astype('float32')).float().to(self.device)

                log_prob = self.flow.log_prob(target_tensor, conditional=conditional_tensor)
            else:
                log_prob = self.flow.log_prob(target_tensor)
        return log_prob.cpu().numpy()
        
    #### DEBUGGIN
    def Flow_posterior(self, target, conditional): 
        self.flow.eval()
        self.flow.to(self.device)
        with torch.no_grad():
            logprobx = self.flow.log_prob(target.to(self.device), conditional=conditional.to(self.device))
            logprobx = logprobx.detach().cpu().numpy() 
            return  logprobx
            
    def p_theta_H0_full_single_14d(self, df, conditional):
        "Functio for evaluating the numerator, takes in all the posterior samples for singular vlaue of the conditional statement"
        "Input: df = Data frame of posterior samples (dl, ra, dec, m1, m2) or (x,y,z, m1, m2)"
        "       conditional = singular value of the conditional statemnt (thsi case, H0 = 70 example ) "
        
        "Output: p(theta|H0,D) numerator "
        
        if 'geocent_time' in df.columns:     
            df.geocent_time = df.geocent_time%86164.0905
            
        conv_df = df    
        # conv_df = self.convert_data(df) #convert data 
        scaled_theta = self.scaler_x.transform(conv_df) #scale data 
        conditional = np.repeat(conditional, len(conv_df))  #repeat condittional singualr value N times as many posterior sampels 
        Y_H0_conditional = self.scaler_y.transform(conditional.reshape(-1,1)) #scael conditional statemnt 
        samples = scaled_theta

        if self.hyperparameters['xyz'] == 0: #check if in sherical or cartesian coordiantes 
            dict_rand = {'luminosity_distance':samples[:,0], 'ra':samples[:,1], 'dec':samples[:,2], 'm1':samples[:,3], 'm2':samples[:,4],
                         'a_1':samples[:,5], 'a_2':samples[:,6],'tilt_1':samples[:,7], 'tilt_2':samples[:,8], 'theta_jn':samples[:,9], 'phi_jl':samples[:,10],
                 'phi_12':samples[:,11], 'psi':samples[:,12],'geocent_time':samples[:,13]}

        # elif flow_class.hyperparameters['xyz'] == 1:
        #     dict_rand = {'x':samples[:,0], 'y':samples[:,1], 'z':samples[:,2],'m1':samples[:,3], 'm2':samples[:,4]}

        samples = pd.DataFrame(dict_rand) #make data frame to pass 
        scaled_theta = samples 
        # if self.hyperparameters['log_it'] == 1:
        #     utilities.logit_data(scaled_theta)
        #     scaled_theta = scaled_theta[np.isfinite(scaled_theta).all(1)]

        scaled_theta = np.array(scaled_theta) 
        scaled_theta = scaled_theta.T*np.ones((1,len(Y_H0_conditional)))

        conditional = np.array(Y_H0_conditional)
        data = np.array(conditional)
        data_scaled = torch.from_numpy(data.astype('float32'))
        Log_Prob = self.Flow_posterior(torch.from_numpy(scaled_theta.T).float(), data_scaled)

        return np.exp(Log_Prob)
        
    
    def convert_data(self, df):
        """
        Convert data from spherical to cartesian coordinates if needed.

        Parameters:
            df (pd.DataFrame): Dataframe containing data to be converted.

        Returns:
            pd.DataFrame: Converted data.
        """
        if self.hyperparameters.get('xyz', 0) == 0:
            return df[['luminosity_distance', 'ra', 'dec', 'mass_1', 'mass_2']]
        else:
            dl = df['luminosity_distance'].to_numpy()
            ra = df['ra'].to_numpy()
            dec = df['dec'].to_numpy()
            x, y, z = cosmology.spherical_to_cart(dl, ra, dec)
            df['x'], df['y'], df['z'] = x, y, z
            return df[['x', 'y', 'z', 'mass_1', 'mass_2']]

    def reshape_conditional(self, conditional, n_samples, axis=1):
        """
        Utility function to reshape conditional input.

        Parameters:
            conditional (np.ndarray): Conditional input.
            n_samples (int): Number of samples to reshape to.
            axis (int, optional): Axis to repeat along. Defaults to 1.

        Returns:
            np.ndarray: Reshaped conditional input.
        """
        conditional = np.repeat(conditional, n_samples, axis=axis)
        return conditional.reshape(n_samples, -1)

    def p_theta_OMEGA_test(self, df, conditional):
        """
        Evaluate log-probabilities for target data given conditional input.

        Parameters:
            df (pd.DataFrame): Dataframe containing target data.
            conditional (list or array): Conditional input values.

        Returns:
            np.ndarray: Log-probability values.
        """
        n_conditional = self.hyperparameters['n_conditional_inputs']
        scaled_theta = self.convert_data(df)
        scaled_theta = self.scaler_x.transform(scaled_theta)
        N_samples = scaled_theta.shape[0]
        conditional = self.scaler_y.transform(conditional.reshape(-1, n_conditional))
        conditional = np.repeat(conditional, N_samples, axis=0)  # Simplified reshaping

        target_tensor = torch.from_numpy(scaled_theta.astype('float32')).float()
        conditional_tensor = torch.from_numpy(conditional.astype('float32'))

        log_prob = self.evaluate_log_prob(target_tensor, conditional_tensor)
        return log_prob.astype('float32')

    def flow_input(self, prior_samples, Nevents, posterior_samples):
        """
        Prepare flow input data from prior and posterior samples.

        Parameters:
            prior_samples (np.ndarray): Prior samples.
            Nevents (int): Number of events.
            posterior_samples (np.ndarray): Posterior samples.

        Returns:
            tuple: Flow data and conditional reshaped for flow input.
        """
        N_prior_values, N_prior_components = prior_samples.shape
        Nevents, N_samples_per_event, N_gw_params = posterior_samples.shape

        # Repeat prior and posterior samples for flow input
        conditional_array_to_flow = np.repeat(prior_samples[:, np.newaxis, np.newaxis, :], Nevents * N_samples_per_event, axis=1)
        data_array_to_flow = np.tile(posterior_samples, (N_prior_values, 1, 1, 1))

        # Reshape for flow input
        flow_data = data_array_to_flow.reshape(N_prior_values * N_samples_per_event * Nevents, N_gw_params)
        flow_conditional = conditional_array_to_flow.reshape(N_prior_values * N_samples_per_event * Nevents, N_prior_components)

        return flow_data, flow_conditional

    def temp_funct_post(self, target_data, prior_samples, N_events, N_post, N_priors, ndim_target=5):
        """
        Evaluate posterior probabilities for the given target and prior samples.

        Parameters:
            target_data (np.ndarray): Target data values.
            prior_samples (np.ndarray): Prior samples.
            N_events (int): Number of events.
            N_post (int): Number of posterior samples.
            N_priors (int): Number of prior samples.
            ndim_target (int, optional): Dimensionality of the target data. Defaults to 5.

        Returns:
            np.ndarray: Log-posterior values.
        """
        scaled_theta = self.scaler_x.transform(target_data)
        conditional = self.scaler_y.transform(prior_samples.T)

        target_tensor, conditional_tensor = self.flow_input(conditional, N_post, scaled_theta.reshape(N_events, N_post, ndim_target))
        target_tensor = torch.from_numpy(target_tensor.astype('float32')).float()
        conditional_tensor = torch.from_numpy(conditional_tensor.astype('float32'))

        log_post = self.evaluate_log_prob(target_tensor, conditional_tensor)
        return self.evaluate_flow_output(log_post, N_events, N_post, N_priors)

    def evaluate_flow_output(self, flow_output_array, Nevents, N_samples_per_event, Npriors):
        """
        Process and reshape flow output into log-posterior probabilities.

        Parameters:
            flow_output_array (np.ndarray): Flow output array.
            Nevents (int): Number of events.
            N_samples_per_event (int): Number of samples per event.
            Npriors (int): Number of priors.

        Returns:
            np.ndarray: Log-posterior probabilities.
        """
        # Reshape flow output
        log_probs_3d = np.reshape(flow_output_array, (Npriors, Nevents, N_samples_per_event), 'C')
        event_prob = np.zeros((Npriors, Nevents))
        sum_over_events = np.zeros((Npriors))

        # Calculate log-posterior for each prior
        for prior in range(Npriors):
            for event in range(Nevents):
                event_prob[prior, event] = logsumexp(log_probs_3d[prior, event, :]) - np.log(len(log_probs_3d[prior, event, :]))
            sum_over_events[prior] = np.sum(event_prob[prior, :])

        return sum_over_events

# Usage example:
# flow_handler = HandleFlow(path='model_path', flow_name='my_flow', device='cpu')
# samples = flow_handler.sample_flow(conditional=[[70]], n_samples=1000)
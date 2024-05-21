import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None 
import h5py
import numpy as np
import sys
sys.path.append("..")
from gw_functions import pdet_theta 
from cosmology_functions import cosmology
from cosmology_functions import utilities 
from gw_functions.gw_SNR_v2 import run_bilby_sim
from tqdm import tqdm
from glasflow.flows import RealNVP, CouplingNSF
import torch 
import pickle 
import corner
import os 
import multiprocessing 
import json
from scipy.spatial.distance import jensenshannon
from scipy import interpolate
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from scipy.stats import ncx2
import bilby

torch.set_printoptions(precision=12)



class Handle_Flow(object):
    def __init__(self, path, flow_name, device, epoch = None, threads = 1, conditional = 1):
        self.path = path
        self.flow_name = flow_name
        self.device = device 
        self.epoch = epoch
        self.threads = threads
        if conditional == 1:
            self.flow, self.hyperparameters, self.scaler_x, self.scaler_y = self.load_hyperparameters_scalers_flow()
        else: 
            self.flow, self.hyperparameters, self.scaler_x = self.load_hyperparameters_scalers_flow()
        
    def load_hyperparameters_scalers_flow(self):
        torch.set_num_threads(self.threads)
        # print(os.getcwd())
        #Open hyperparameter dictionary
        path = self.path
        flow_name = self.flow_name
        hyper = open(path+flow_name+'/'+'hyperparameters.txt', 'r').read()
        hyperparameters = eval(hyper)

        device = 'cpu'
        n_inputs = hyperparameters['n_inputs']
        n_conditional_inputs = hyperparameters['n_conditional_inputs'] 
        n_neurons = hyperparameters['n_neurons']
        n_transforms = hyperparameters['n_transforms']
        n_blocks_per_transform = hyperparameters['n_blocks_per_transform']
        dropout = hyperparameters['dropout']
        flow_type = hyperparameters['flow_type']

        #open scaler_x and scaler_y
        scalerfile_x = path+flow_name+'/'+'scaler_x.sav'
        scaler_x = pickle.load(open(scalerfile_x, 'rb'))

        if n_conditional_inputs != 0:
            scalerfile_y = path+flow_name+'/'+'scaler_y.sav'
            scaler_y = pickle.load(open(scalerfile_y, 'rb'))


        #Open flow model file flow.pt
        if self.epoch is None: 
            flow_load = torch.load(path+flow_name+'/'+'flow.pt', map_location=self.device)
        else: 
            flow_load = torch.load(path+flow_name+'/flows_epochs/'+'flow_epoch_{}.pt'.format(self.epoch), map_location=self.device)
            

        if flow_type == 'RealNVP':
            flow_empty = RealNVP(n_inputs= n_inputs,
                n_transforms= n_transforms,
                n_neurons= n_neurons,
                n_conditional_inputs = n_conditional_inputs,
                n_blocks_per_transform = n_blocks_per_transform,
                batch_norm_between_transforms=True,
                # batch_norm_within_blocks=True,
                dropout_probability=dropout,
                linear_transform='lu',
                volume_preserving = volume_preserving)
        elif flow_type == 'CouplingNSF':   
                flow_empty = CouplingNSF(n_inputs= n_inputs,
                n_transforms= n_transforms,
                n_neurons= n_neurons,
                n_conditional_inputs = n_conditional_inputs,
                n_blocks_per_transform = n_blocks_per_transform,
                batch_norm_between_transforms=True,
                dropout_probability=dropout,
                linear_transform='lu')

        flow_empty.load_state_dict(flow_load)
        flow = flow_empty
        flow.eval()

        if n_conditional_inputs != 0:
            return flow, hyperparameters, scaler_x, scaler_y
        else:
            return flow, hyperparameters, scaler_x
    
    
    def Flow_samples(self, conditional, n):
        "Sample the flow using conditional statements"
        n_conditional = self.hyperparameters['n_conditional_inputs']
        conditional = np.array(conditional).T
        conditional = self.scaler_y.transform(conditional.reshape(-1,n_conditional)) #scael conditional statemnt
        
        # print(np.shape(self.scaler_y.inverse_transform(conditional)))
        # print(self.scaler_y.inverse_transform(conditional))
        
        data = np.array(conditional)
        
        data_scaled = torch.from_numpy(data.astype('float32'))
        self.flow.eval()
        self.flow.to('cpu')
        with torch.no_grad():
            samples = self.flow.sample(n, conditional=data_scaled.to('cpu'))
            samples= self.scaler_x.inverse_transform(samples.to('cpu'))
        return samples
    

    def Flow_posterior(self, target, conditional): 
        self.flow.eval()
        self.flow.to(self.device)
        with torch.no_grad():
            logprobx = self.flow.log_prob(target.to(self.device), conditional=conditional.to(self.device))
            logprobx = logprobx.detach().cpu().numpy() 
            return  logprobx

    def Flow_posterior_no_conditional(self, target): 
        self.flow.eval()
        self.flow.to(self.device)
        with torch.no_grad():
            logprobx = self.flow.log_prob(target.to(self.device))
            logprobx = logprobx.detach().cpu().numpy() 
            return  logprobx
 
        
    def convert_data(self, df):
        data = df
        coordinates= data[['luminosity_distance', 'ra', 'dec']]
        dl = np.array(coordinates.luminosity_distance)
        ra = np.array(coordinates.ra)
        dec = np.array(coordinates.dec)
      

        if self.hyperparameters['xyz'] == 0:
            return data[['luminosity_distance', 'ra', 'dec','mass_1', 'mass_2']]
        
        if self.hyperparameters['xyz'] == 1:
            x, y, z = cosmology.spherical_to_cart(dl, ra, dec)
            data['x'] = x
            data['y'] = y
            data['z'] = z
            return data[['xcoord', 'ycoord', 'zcoord','mass_1', 'mass_2']]
    
    def p_theta_OMEGA_test(self, df, conditional):
        n_conditional = self.hyperparameters['n_conditional_inputs'] ### Get N conditional
        scaled_theta = self.convert_data(df) #convert data 
        # print(scaled_theta, df)
        scaled_theta = self.scaler_x.transform(scaled_theta) #scale data 
        scaled_theta = np.array(scaled_theta) ## make sure it is an array 
        N_samples = np.shape(scaled_theta)[0] ## get the Number of target data rows
        conditional = self.scaler_y.transform(conditional.reshape(-1,n_conditional)) ##scale conditional, and reshape 
        # print(conditional)
        # print(scaled_theta)
        # print(conditional)
        # conditional = np.repeat(conditional, N_samples) ### Repeat conditional to match the nubmer of target rows
        conditional = np.repeat(conditional, N_samples, axis = 1) ### Repeat conditional to match the nubmer of target rows
        # print(np.shape(conditional))
        # print(conditional)
        conditional = conditional.reshape(n_conditional, N_samples).T## Reshape accordingly GOOD
        # conditional = conditional.T.reshape(N_samples, n_conditional) ## Reshape accordingly 
        # conditional = conditional.reshape(N_samples, n_conditional) ## Reshape accordingly GOOD
        # conditional = conditional.reshape(n_conditional, N_samples).T ## Reshape accordingly 
        # print(conditional)
        
        target_tensor = torch.from_numpy(scaled_theta.astype('float32')).float() 
        conditional_tensor = torch.from_numpy(conditional.astype('float32'))
        
        # print(target_tensor.shape, conditional_tensor.shape)
        # print(target_tensor, conditional_tensor)

        Log_Prob = self.Flow_posterior(target_tensor, conditional_tensor)
        return Log_Prob.astype('float32')
    
    
    
    def p_theta_OMEGA_test_v2(self, df, conditional):
        n_conditional = self.hyperparameters['n_conditional_inputs'] ### Get N conditional
        scaled_theta = self.convert_data(df) #convert data 
        # print(scaled_theta, df)
        scaled_theta = self.scaler_x.transform(scaled_theta) #scale data 
        scaled_theta = np.array(scaled_theta) ## make sure it is an array 
        N_samples = np.shape(scaled_theta)[0] ## get the Number of target data rows
        conditional = self.scaler_y.transform(conditional.reshape(-1,n_conditional)) ##scale conditional, and reshape 
        # print(conditional)
        # print(scaled_theta)
        # print(conditional)
        # conditional = np.repeat(conditional, N_samples) ### Repeat conditional to match the nubmer of target rows
        conditional = np.repeat(conditional, N_samples, axis = 1) ### Repeat conditional to match the nubmer of target rows
        print(np.shape(conditional))
        # print(conditional)
        # conditional = conditional.T.reshape(N_samples, n_conditional) ## Reshape accordingly 
        # conditional = conditional.reshape(N_samples, n_conditional) ## Reshape accordingly GOOD
        # conditional = conditional.reshape(n_conditional, N_samples).T ## Reshape accordingly 
        # print(conditional)

        target_tensor = torch.from_numpy(scaled_theta.astype('float32')).float() 
        conditional_tensor = torch.from_numpy(conditional.astype('float32'))

        # print(target_tensor.shape, conditional_tensor.shape)
        # print(target_tensor, conditional_tensor)

        Log_Prob = self.Flow_posterior(target_tensor, conditional_tensor)
        return Log_Prob.astype('float32')

    
    def p_theta_OMEGA_test_batching(self, df, conditional):
        n_conditional = self.hyperparameters['n_conditional_inputs']
        conditional = conditional.T
        N_priors = np.shape(conditional)[0]
        N_samples = len(df) ## check how many rows the data has
        df = pd.concat([df]*N_priors, ignore_index=True) ### start batching by repeating the dataframe Nprior times
        
        # print(df)
        scaled_theta = self.convert_data(df) #convert data 
        scaled_theta = self.scaler_x.transform(scaled_theta) #scale data 
        scaled_theta = np.array(scaled_theta) ### make sure the data is an array
        conditional = self.scaler_y.transform(conditional.reshape(-1,n_conditional))  ### reshape conditional data 
        # print(conditional)
        # conditional = np.repeat(conditional, N_samples, axis = 0)
        # print(np.shape(conditional))
        # print(conditional)
        
        
        # conditional = (np.repeat(conditional, N_samples).reshape(int(N_samples*N_priors), n_conditional ))
        # print(conditional[:,0])
        # print(scaled_theta[0,:])
        conditional = (np.repeat(conditional, N_samples).reshape(n_conditional, int(N_samples*N_priors))).T 
        ## repeat conditional to match NtargetxNprior

        target_tensor = torch.from_numpy(scaled_theta.astype('float32')).float()
        conditional_tensor = torch.from_numpy(conditional.astype('float32'))
        
        # print(target_tensor, conditional_tensor)
        Log_Prob = self.Flow_posterior(target_tensor, conditional_tensor)
        return Log_Prob.astype('float32')
  

    def p_theta_OMEGA_test_batching_Chris(self, df, conditional):
        n_conditional = self.hyperparameters['n_conditional_inputs']
        conditional = conditional.T
        N_priors = np.shape(conditional)[0]
        N_samples = len(df) ## check how many rows the data has
        
        
        df = pd.concat([df]*N_priors, ignore_index=True) ### start batching by repeating the dataframe Nprior times

        # print(df)
        scaled_theta = self.convert_data(df) #convert data 
        scaled_theta = self.scaler_x.transform(scaled_theta) #scale data 
        scaled_theta = np.array(scaled_theta) ### make sure the data is an array
        conditional = self.scaler_y.transform(conditional.reshape(-1,n_conditional))  ### reshape conditional data 
        conditional = (np.repeat(conditional, N_samples).reshape(int(N_samples*N_priors),n_conditional)) 
        ## repeat conditional to match NtargetxNprior

        target_tensor = torch.from_numpy(scaled_theta.astype('float32')).float()
        conditional_tensor = torch.from_numpy(conditional.astype('float32'))

        # print(target_tensor, conditional_tensor)
        Log_Prob = self.Flow_posterior(target_tensor, conditional_tensor)
        return Log_Prob.astype('float32')
  

    def flow_input(self, prior_samples, Nevents, posterior_samples):
        
        N_prior_values, N_prior_components = np.shape(prior_samples)

        # data array
        Nevents, N_samples_per_event, N_gw_params = np.shape(posterior_samples)
        
        
        # print(N_prior_values, N_prior_components , Nevents, N_samples_per_event, N_gw_params)
        # initialise empty conditional array
        conditional_array_to_flow = np.zeros((N_prior_values, Nevents, N_samples_per_event, N_prior_components))

        # initialise empty data array
        data_array_to_flow = np.zeros((N_prior_values, Nevents, N_samples_per_event, N_gw_params))

        # loop through EOSs
        for i in range(0, N_prior_values):

            # loop through events (1 in this case)
            for j in range(0, Nevents):

                # loop through samples in each event
                for k in range(0, N_samples_per_event):

                    data_array_to_flow[i, j, k, :] = posterior_samples[j, k, :]

                    conditional_array_to_flow[i, j, k, :] = prior_samples[i, :]
                    
        flow_data = np.reshape(data_array_to_flow, (N_prior_values, N_samples_per_event*Nevents, N_gw_params))

        # conditional array
        flow_conditional = np.reshape(conditional_array_to_flow, (N_prior_values, N_samples_per_event*Nevents, N_prior_components))
                    
            
        if Nevents == 1:
            flow_data = np.reshape(flow_data, (N_prior_values*N_samples_per_event*Nevents, N_gw_params))
            flow_conditional = np.reshape(flow_conditional, (N_prior_values*N_samples_per_event*Nevents, N_prior_components))

            # # check that the final reshape is successful
            # if (data_array_to_flow[5,:]).all() == (flow_data[-1,:]).all() and (conditional_array_to_flow[5,:]).all() == (flow_conditional[-1,:]).all():
            #     print('Batch for flow made successfully; reshaped into\n', 'data:', np.shape(flow_data), 'conditional', np.shape(flow_conditional))

        else:

            # testing reshaping is as expected
            if (data_array_to_flow[0,1,0,:]).all() == (flow_data[0,-1,:]).all() and (conditional_array_to_flow[0,1,0,:]).all() == (flow_conditional[0,-1,:]).all():
                flow_data = np.reshape(flow_data, (N_prior_values*N_samples_per_event*Nevents, N_gw_params))
                flow_conditional = np.reshape(flow_conditional, (N_prior_values*N_samples_per_event*Nevents, N_prior_components))
            else:
                print('arrays reshaped incorrectly')


            
          
        return flow_data, flow_conditional
    
    
    
    def temp_funct_post(self, flow_class, target_data, prior_samples, N_events, N_post, N_priors, ndim_target = 5):

        target_data = self.convert_data(target_data) #convert data #####UNCOMMENT!!!!
        scaled_theta = self.scaler_x.transform(target_data) #scale data 
        scaled_theta = np.array(scaled_theta) ### make sure the data is an array
        conditional = self.scaler_y.transform(prior_samples.T)  ### reshape conditional data 

        # print(prior_samples, conditional)

        target_tensor , conditional_tensor = self.flow_input(conditional, N_post , np.array(scaled_theta).reshape(N_events,N_post, ndim_target)) ### Change 1 to 5

        target_tensor = torch.from_numpy(target_tensor.astype('float32')).float()
        conditional_tensor = torch.from_numpy(conditional_tensor.astype('float32'))


        # print(np.shape(target_tensor), np.shape(conditional_tensor))
        # print(target_tensor, conditional_tensor)
        log_post = self.Flow_posterior(target_tensor, conditional_tensor)
        log_post = self.evaluate_flow_output(log_post, N_events, N_post, N_priors)
        return log_post
    
    
    
    def evaluate_flow_output(self, flow_output_array, Nevents, N_samples_per_event, Npriors):
                # take in the data from the flow
        # the number of log probs need to be divided up into the correct number
        # per PCA, per event, each sample

        # if 1 event, keep things in 2d, and then just pad with an extra dimension for the flow

        # number of log probs out of the flow
        N_log_probs = len(flow_output_array)

        # plt.hist(flow_output_array, bins = 1000)
        # plt.savefig(outdir + 'log_probs_from_flow.pdf')
        # plt.close()

        # reshape to 2D, collating samples from all events
        log_probs_2d = np.reshape(flow_output_array, (Npriors, N_samples_per_event*Nevents), 'C')

        # reshape back into 3D
        log_probs_3d = np.reshape(flow_output_array, (Npriors, Nevents, N_samples_per_event), 'C')

        # placeholders for looping
        # log probs
        event_prob = np.zeros((Npriors, Nevents))
        # log posterior
        sum_over_events = np.zeros((Npriors))

        # placeholders for function outputs
        log_likelihoods = []
        EOSs = []

        # loop through EOS locations
        for prior in range(0,Npriors):

            # loop through events
            for event in range(0,Nevents):

                # log sum exp over samples in an event and take log mean
                # i.e. subtract the log number of samples
                event_prob[prior, event] = logsumexp(log_probs_3d[prior, event,:]) - np.log(len(log_probs_3d[prior, event,:]))
                # prior on GW samples is uniform

            # sum over log_probs from all events for a given EOS
            # we have 1 event here
            sum_over_events[prior] = np.sum(event_prob[prior, :])  

        # log posterior on EOS is log likelihood plus log prior
        log_posterior = sum_over_events

        return log_posterior

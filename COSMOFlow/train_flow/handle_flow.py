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

from scipy.stats import ncx2
import bilby



class Handle_Flow(object):
    def __init__(self, path, flow_name, device, epoch = None, threads = 1):
        self.path = path
        self.flow_name = flow_name
        self.device = device 
        self.epoch = epoch
        self.threads = threads
        self.flow, self.hyperparameters, self.scaler_x, self.scaler_y = self.load_hyperparameters_scalers_flow()
        
        
    


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
        scalerfile_y = path+flow_name+'/'+'scaler_y.sav'
        scaler_x = pickle.load(open(scalerfile_x, 'rb'))
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
                dropout_probability=dropout,
                linear_transform='lu')
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

        return flow, hyperparameters, scaler_x, scaler_y
    
    
        
        
        
    def convert_data(self, df):
        data = df

        coordinates= data[['luminosity_distance', 'ra', 'dec']]
        dl = np.array(coordinates.luminosity_distance)
        ra = np.array(coordinates.ra)
        dec = np.array(coordinates.dec)

        x, y, z = cosmology.spherical_to_cart(dl, ra, dec)
        df['x'] = x
        df['y'] = y
        df['z'] = z

        if self.hyperparameters['xyz'] == 0:
            return data[['luminosity_distance', 'ra', 'dec','mass_1', 'mass_2']]
        if self.hyperparameters['xyz'] == 1:
            return data[['xcoord', 'ycoord', 'zcoord','mass_1', 'mass_2']]



    def p_theta_H0(self, df, conditional):
        conv_df = self.convert_data(df)
        scaled_theta = self.scaler_x.transform(np.array(conv_df).reshape(1,-1))
        Y_H0_conditional = self.scaler_y.transform(conditional.reshape(-1,1))
        samples = scaled_theta[0]
        
        if self.hyperparameters['xyz'] == 0:
            dict_rand = {'luminosity_distance':samples[0], 'ra':samples[1], 'dec':samples[2], 'm1':samples[3], 'm2':samples[4]}

        elif self.hyperparameters['xyz'] == 1:
            dict_rand = {'x':samples[0], 'y':samples[1], 'z':samples[2],'m1':samples[3], 'm2':samples[4]}

        samples = pd.DataFrame(dict_rand, index = [0])
        scaled_theta = samples 
        # if self.hyperparameters['log_it'] == 1:
        #     utilities.logit_data(scaled_theta)
        #     scaled_theta = scaled_theta[np.isfinite(scaled_theta).all(1)]

        scaled_theta = np.array(scaled_theta)
        scaled_theta = scaled_theta.T*np.ones((1,len(Y_H0_conditional)))

        conditional = np.array(Y_H0_conditional)
        data = np.array(conditional)
        data_scaled = torch.from_numpy(data.astype('float32'))

        def Flow_posterior(target, conditional, device = 'cpu'): 
            self.flow.eval()
            self.flow.to(self.device)

            with torch.no_grad():

                logprobx = self.flow.log_prob(target.to(device), conditional=conditional.to(device))
                logprobx = logprobx.numpy() 

                return  logprobx

        Log_Prob = Flow_posterior(torch.from_numpy(scaled_theta.T).float(), data_scaled)


        return np.exp(Log_Prob)
    
    
    
    def p_theta_H0_full_single(self, df, conditional):
        "Functio for evaluating the numerator, takes in all the posterior samples for singular vlaue of the conditional statement"
        "Input: df = Data frame of posterior samples (dl, ra, dec, m1, m2) or (x,y,z, m1, m2)"
        "       conditional = singular value of the conditional statemnt (thsi case, H0 = 70 example ) "
        
        "Output: p(theta|H0,D) numerator "
                
        conv_df = self.convert_data(df) #convert data 
        scaled_theta = self.scaler_x.transform(conv_df) #scale data 
        conditional = np.repeat(conditional, len(conv_df))  #repeat condittional singualr value N times as many posterior sampels 
        Y_H0_conditional = self.scaler_y.transform(conditional.reshape(-1,1)) #scael conditional statemnt 
        samples = scaled_theta

        if self.hyperparameters['xyz'] == 0: #check if in sherical or cartesian coordiantes 
            dict_rand = {'luminosity_distance':samples[:,0], 'ra':samples[:,1], 'dec':samples[:,2], 'm1':samples[:,3], 'm2':samples[:,4]}

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

        def Flow_posterior(target, conditional): 
            self.flow.eval()
            self.flow.to(self.device)

            with torch.no_grad():

                logprobx = self.flow.log_prob(target.to(self.device), conditional=conditional.to(self.device))
                logprobx = logprobx.detach().cpu().numpy() 

                return  logprobx

        Log_Prob = Flow_posterior(torch.from_numpy(scaled_theta.T).float(), data_scaled)


        return np.exp(Log_Prob)
    
    
    
    def p_theta_OMEGA(self, df, conditional):
        "Functio for evaluating the numerator, takes in all the posterior samples for singular vlaue of the conditional statement"
        "Input: df = Data frame of posterior samples (dl, ra, dec, m1, m2) or (x,y,z, m1, m2)"
        "       conditional = singular value of the conditional statemnt (thsi case, H0 = 70 example ) "

        "Output: p(theta|H0,D) numerator "
        n_conditional = self.hyperparameters['n_conditional_inputs']
        conv_df = self.convert_data(df) #convert data 
        scaled_theta = self.scaler_x.transform(conv_df) #scale data 
        conditional = np.repeat(conditional, len(conv_df))  #repeat condittional singualr value N times as many posterior sampels 
        Y_conditional = self.scaler_y.transform(conditional.reshape(-1,n_conditional)) #scael conditional statemnt 
        samples = scaled_theta 

        if self.hyperparameters['xyz'] == 0: #check if in sherical or cartesian coordiantes 
            dict_rand = {'luminosity_distance':samples[:,0], 'ra':samples[:,1], 'dec':samples[:,2], 'm1':samples[:,3], 'm2':samples[:,4]}

        elif flow_class.hyperparameters['xyz'] == 1:
            dict_rand = {'x':samples[:,0], 'y':samples[:,1], 'z':samples[:,2],'m1':samples[:,3], 'm2':samples[:,4]}

        samples = pd.DataFrame(dict_rand) #make data frame to pass
        scaled_theta = samples 
        # if self.hyperparameters['log_it'] == 1:
        #     utilities.logit_data(scaled_theta)
        #     scaled_theta = scaled_theta[np.isfinite(scaled_theta).all(1)]

        scaled_theta = np.array(scaled_theta) 
        scaled_theta = scaled_theta.T*np.ones((1,len(Y_conditional)))

        conditional = np.array(Y_conditional)
        data = np.array(conditional)
        data_scaled = torch.from_numpy(data.astype('float32'))

        def Flow_posterior(target, conditional): 
            self.flow.eval()
            self.flow.to(self.device)

            with torch.no_grad():

                logprobx = self.flow.log_prob(target.to(self.device), conditional=conditional.to(self.device))
                logprobx = logprobx.detach().cpu().numpy() 

                return  logprobx

        Log_Prob = Flow_posterior(torch.from_numpy(scaled_theta.T).float(), data_scaled)


        return np.exp(Log_Prob)
    
    
    def Flow_samples(self, conditional, n):
        "Sample the flow using conditional statements"
        n_conditional = self.hyperparameters['n_conditional_inputs']
        Y_conditional = self.scaler_y.transform(conditional.reshape(-1,n_conditional)) #scael conditional statemnt 


        conditional = np.array(Y_conditional)
        data = np.array(conditional)
        data_scaled = torch.from_numpy(data.astype('float32'))

        self.flow.eval()
        self.flow.to('cpu')


        with torch.no_grad():
            samples = self.flow.sample(n, conditional=data_scaled.to('cpu'))
            samples= self.scaler_x.inverse_transform(samples.to('cpu'))

        return samples
    
    
    
    
    def p_theta_H0_one_go_batch(self, df, conditional):
        "Functio for evaluating the numerator, takes in all the posterior samples for batch of the conditional statement"
        "Input: df = Data frame of posterior samples (dl, ra, dec, m1, m2) or (x,y,z, m1, m2)"
        "       conditional = singular value of the conditional statemnt (thsi case, H0 = 70 example ) "
        "Output: p(theta|H0,D) numerator "
        
        N_samples = len(conditional); N_theta = len(df)
        
        conv_df = self.convert_data(df) #convert data 
        scaled_theta = self.scaler_x.transform(conv_df) #scale data 
        # conditional = np.repeat(conditional, len(conv_df))  #repeat condittional singualr value N times as many posterior sampels 
        Y_H0_conditional = self.scaler_y.transform(conditional.reshape(-1,self.hyperparameters['n_conditional_inputs'])) #scael conditional statemnt 
        samples = scaled_theta

        if self.hyperparameters['xyz'] == 0: #check if in sherical or cartesian coordiantes 
            dict_rand = {'luminosity_distance':samples[:,0], 'ra':samples[:,1], 'dec':samples[:,2], 'm1':samples[:,3], 'm2':samples[:,4]}

        # elif flow_class.hyperparameters['xyz'] == 1:
        #     dict_rand = {'x':samples[:,0], 'y':samples[:,1], 'z':samples[:,2],'m1':samples[:,3], 'm2':samples[:,4]}

        samples = pd.DataFrame(dict_rand) #make data frame to pass 
        scaled_theta = samples 
        scaled_theta = np.array(scaled_theta) 
        conditional = np.array(Y_H0_conditional)
        
        
        target_tensor, conditional_tensor = self.make_data_batch_flow(scaled_theta, conditional, N_samples)
        target_tensor = torch.from_numpy(target_tensor).float()
        conditional_tensor = torch.from_numpy(conditional_tensor.astype('float32'))
        
        # print(np.shape(target_tensor), np.shape(conditional_tensor))
        def Flow_posterior(target, conditional): 
            self.flow.eval()
            self.flow.to(self.device)
            with torch.no_grad():
                logprobx = self.flow.log_prob(target.to(self.device), conditional=conditional.to(self.device))
                logprobx = logprobx.detach().cpu().numpy() 
                return  logprobx

        Log_Prob = Flow_posterior(target_tensor, conditional_tensor)
        return np.exp(Log_Prob)
    
    
    
    
    def make_data_batch_flow(self, target, conditional, N_samples):
        
        N_theta = len(target)
        data_tensor = np.zeros((N_theta,N_samples,self.hyperparameters['n_inputs']))
        conditional_tensor = np.zeros((N_theta,N_samples, self.hyperparameters['n_conditional_inputs']))
        
        for i in range(N_samples):
            data_tensor[:,i,:] = target
            conditional_tensor[:,i,:] = conditional[i]
            
        data_tensor = data_tensor.reshape(int(N_theta*N_samples), self.hyperparameters['n_inputs'])
        conditional_tensor = conditional_tensor.reshape(int(N_theta*N_samples), self.hyperparameters['n_conditional_inputs'])
        
        return data_tensor, conditional_tensor
    
    def get_posterior_from_batch(self, flow_samples ,N_samples,  N_theta , N_event):
        log_posterior = np.ones(N_samples)
        for k in range(N_event):
            if k == N_event - 1: 
                break
            flow_samples_temp = flow_samples[int(k*(N_theta)):int((k+1)*N_theta), : ]
            log_posterior += np.log(np.sum(flow_samples_temp,axis = 0)/N_theta )
        posterior = np.exp(log_posterior)
        return posterior
        

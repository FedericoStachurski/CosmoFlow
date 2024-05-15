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

torch.set_printoptions(precision=12)



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
        volume_preserving = hyperparameters['volume_preserving']

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

        return flow, hyperparameters, scaler_x, scaler_y
    
    
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
    

    def Flow_posterior(self, target, conditional): 
        self.flow.eval()
        self.flow.to(self.device)
        with torch.no_grad():
            logprobx = self.flow.log_prob(target.to(self.device), conditional=conditional.to(self.device))
            logprobx = logprobx.detach().cpu().numpy() 
            return  logprobx
        
        
        
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
            return df[['luminosity_distance', 'ra', 'dec','mass_1', 'mass_2']]
        if self.hyperparameters['xyz'] == 1:
            return df[['xcoord', 'ycoord', 'zcoord','mass_1', 'mass_2']]



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


        Log_Prob = self.Flow_posterior(torch.from_numpy(scaled_theta.T).float(), data_scaled)


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
        Log_Prob = self.Flow_posterior(torch.from_numpy(scaled_theta.T).float(), data_scaled)

        return np.exp(Log_Prob)
    
    def p_theta_OMEGA_test(self, df, conditional):
        n_conditional = self.hyperparameters['n_conditional_inputs']
        scaled_theta = self.convert_data(df) #convert data 
        scaled_theta = self.scaler_x.transform(scaled_theta) #scale data 
        scaled_theta = np.array(scaled_theta)
        N_samples = np.shape(scaled_theta)[0]
        conditional = self.scaler_y.transform(conditional.reshape(-1,n_conditional)) 
 
        conditional = np.repeat(conditional, N_samples)
        conditional = conditional.reshape(N_samples, n_conditional)

        target_tensor = torch.from_numpy(scaled_theta.astype('float32')).float()
        conditional_tensor = torch.from_numpy(conditional.astype('float32'))
        
        
        # print(target_tensor, conditional_tensor)
        Log_Prob = self.Flow_posterior(target_tensor, conditional_tensor)
        return Log_Prob.astype('float32')
    
    
    def p_theta_OMEGA_test_batching(self, df, conditional):
        n_conditional = self.hyperparameters['n_conditional_inputs']
        conditional = conditional.T
        N_priors = np.shape(conditional)[0]
        N_samples = len(df)
        df = pd.concat([df]*N_priors, ignore_index=True)
        
        # print(df)
        scaled_theta = self.convert_data(df) #convert data 
        scaled_theta = self.scaler_x.transform(scaled_theta) #scale data 
        scaled_theta = np.array(scaled_theta)
        conditional = self.scaler_y.transform(conditional.reshape(-1,n_conditional)) 
        conditional = (np.repeat(conditional, N_samples).reshape(int(N_samples*N_priors),n_conditional))

        target_tensor = torch.from_numpy(scaled_theta.astype('float32')).float()
        conditional_tensor = torch.from_numpy(conditional.astype('float32'))
        
        # print(target_tensor, conditional_tensor)
        Log_Prob = self.Flow_posterior(target_tensor, conditional_tensor)
        return Log_Prob.astype('float32')
  
    
    
    
    
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
        
        # scaled_theta = scaled_theta.T*np.ones((1,len(Y_conditional)))
        # print(np.shape(scaled_theta))
        
        conditional = np.array(Y_conditional)
        data = np.array(conditional)
        data_scaled = torch.from_numpy(data.astype('float32'))
        Log_Prob = self.Flow_posterior(torch.from_numpy(scaled_theta).float(), data_scaled)

        return Log_Prob
    
    

    
    def p_theta_H0_one_go_batch_DETECTORS(self, df, conditional):
        "Functio for evaluating the numerator, takes in all the posterior samples for batch of the conditional statement"
        "Input: df = Data frame of posterior samples (dl, ra, dec, m1, m2) or (x,y,z, m1, m2)"
        "       conditional = singular value of the conditional statemnt (thsi case, H0 = 70 example ) "
        "Output: p(theta|H0,D) numerator "
        
        N_samples = len(conditional); N_theta = len(df)
        
        det_set = df[['H1', 'L1', 'V1']] ########NEW 
        conv_df = self.convert_data(df) #convert data 

        scaled_theta = self.scaler_x.transform(conv_df) #scale data 
        conditional = np.hstack((conditional, np.ones((len(conditional),3)))) ########NEW 

        Y_H0_conditional = self.scaler_y.transform(conditional.reshape(-1,self.hyperparameters['n_conditional_inputs'])) #scael conditional statemnt 
        samples = scaled_theta
        if self.hyperparameters['xyz'] == 0: #check if in sherical or cartesian coordiantes 
            dict_rand = {'luminosity_distance':samples[:,0], 'ra':samples[:,1], 'dec':samples[:,2], 'm1':samples[:,3], 'm2':samples[:,4]}

        elif self.hyperparameters['xyz'] == 1:
            dict_rand = {'x':samples[:,0], 'y':samples[:,1], 'z':samples[:,2],'m1':samples[:,3], 'm2':samples[:,4]}
        samples = pd.DataFrame(dict_rand) #make data frame to pass 
        scaled_theta = samples 
        scaled_theta = np.array(scaled_theta) 
        conditional = np.array(Y_H0_conditional)[:, :-3] ########NEW 
        target_tensor, conditional_tensor = self.make_data_batch_flow_DETECTORS(scaled_theta, conditional, N_samples, det_set) ########NEW 
        target_tensor = torch.from_numpy(target_tensor).float()
        conditional_tensor = torch.from_numpy(conditional_tensor.astype('float32'))

        Log_Prob = self.Flow_posterior(target_tensor, conditional_tensor)
        return Log_Prob
    
    def make_data_batch_flow_DETECTORS(self, target, conditional, N_samples, det_setup):
        # det_setup = np.array(target)[:,-3:]
        det_setup = np.array(det_setup)
        target = np.array(target)

        N_theta = len(target)
        data_tensor = np.zeros((N_theta,N_samples,len(target[0])))

        
        conditional_tensor = np.zeros((N_theta,N_samples,len(conditional[0,:])))
        setup_tensor = np.zeros((N_theta,N_samples,3))

        for i in range(N_samples):
            data_tensor[:,i,:] = target
            conditional_tensor[:,i,:] = conditional[i,:]
            setup_tensor[:,i,:] = det_setup
        cond_setup_tensor = np.concatenate([conditional_tensor, setup_tensor], -1)
        data_tensor = data_tensor.reshape(int(N_theta*N_samples), len(target[0]))
        cond_setup_tensor = cond_setup_tensor.reshape(int(N_theta*N_samples),len(conditional[0,:])+3)

        return data_tensor, cond_setup_tensor
    
    
    
    def p_theta_H0_one_go_batch(self, df, conditional):
        "Functio for evaluating the numerator, takes in all the posterior samples for batch of the conditional statement"
        "Input: df = Data frame of posterior samples (dl, ra, dec, m1, m2) or (x,y,z, m1, m2)"
        "       conditional = singular value of the conditional statemnt (thsi case, H0 = 70 example ) "
        "Output: p(theta|H0,D) numerator "
        
        N_samples = len(conditional); N_theta = len(df)
        conv_df = self.convert_data(df) #convert data 
        scaled_theta = self.scaler_x.transform(conv_df) #scale data 
        Y_H0_conditional = self.scaler_y.transform(conditional.reshape(-1,self.hyperparameters['n_conditional_inputs'])) #scael conditional statemnt 
        samples = scaled_theta
        if self.hyperparameters['xyz'] == 0: #check if in sherical or cartesian coordiantes 
            dict_rand = {'luminosity_distance':samples[:,0], 'ra':samples[:,1], 'dec':samples[:,2], 'm1':samples[:,3], 'm2':samples[:,4]}

        elif self.hyperparameters['xyz'] == 1:
            dict_rand = {'x':samples[:,0], 'y':samples[:,1], 'z':samples[:,2],'m1':samples[:,3], 'm2':samples[:,4]}

        samples = pd.DataFrame(dict_rand) #make data frame to pass 
        scaled_theta = samples 
        scaled_theta = np.array(scaled_theta) 
        conditional = np.array(Y_H0_conditional)
        print(np.shape(conditional))
        target_tensor, conditional_tensor = self.make_data_batch_flow(scaled_theta, conditional, N_samples) ########NEW 
        target_tensor = torch.from_numpy(target_tensor).float()
        conditional_tensor = torch.from_numpy(conditional_tensor.astype('float32'))

        Log_Prob = self.Flow_posterior(target_tensor, conditional_tensor)
        return Log_Prob
    
    
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
        epsilon = 1e-25
        
        log_posterior = np.ones(N_samples)
        for k in range(N_event):
            if k == N_event - 1: 
                break
            flow_samples_temp = flow_samples[int(k*(N_theta)):int((k+1)*N_theta), : ]
            
            log_posterior += np.log(np.sum(np.exp(flow_samples_temp+epsilon),axis = 0)/N_theta)
        
        return log_posterior
    
    def get_posterior_from_batch_JESS(self, flow_samples ,N_samples,  N_theta , N_event):
        # epsilon = 1e-25

        # log_likelihood = np.zeros((len(conditional_samples.T)))
        log_likelihood = np.ones((N_samples, 1))
        
        for _ in range(N_samples):
            for k in range(N_event):
                if k == N_event - 1: 
                    break
                # print(np.shape(flow_samples))
                flow_samples_temp = flow_samples[int(k*(N_theta)):int((k+1)*N_theta), : ]
                # print(np.shape(flow_samples_temp)) #1x100
                # print(np.shape(np.log(np.sum(np.exp(flow_samples_temp)+epsilon,axis = 0)/N_theta)))
                log_likelihood += np.log(np.sum(np.exp(flow_samples_temp),axis = 1)/N_theta)
                # print(np.sum(flow_samples_temp))
                # print(np.shape(flow_samples_temp))
                # print(np.sum(flow_samples_temp))
                # print(np.shape(flow_samples_temp))
                # log_likelihood += np.log(np.exp(flow_samples_temp.T)/np.sum(np.exp(flow_samples_temp.T)))
        # print(log_likelihood)
        # log_likelihood = np.sum(log_likelihood, axis = 0)
        # print(np.shape(log_likelihood))

        return log_likelihood
        

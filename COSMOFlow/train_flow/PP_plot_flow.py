from glasflow.flows import RealNVP, CouplingNSF
from scipy.stats import norm
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import display
from tqdm import tqdm 
import corner
import numpy as np
import pickle
from scipy.special import erf
from cosmology_functions import priors 
from cosmology_functions import cosmology 
import astropy.constants as const
from scipy.stats import norm, gaussian_kde
from tqdm import tqdm 
from scipy.stats import ncx2
import sys
import os
import shutil
import pickle
import bilby
from bilby.core.prior import Uniform
from scipy.stats import entropy
bilby.core.utils.log.setup_logger(log_level=0)
from astropy.coordinates import spherical_to_cartesian, cartesian_to_spherical
from torch import logit, sigmoid

import argparse


# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-Name", "--Name_folder", required=True,
   help="Name of the folder to save the FLOW model")
ap.add_argument("-data", "--data_folder", required=True,
   help="Name of the folder where training data is stored")





args = vars(ap.parse_args())
Name = str(args['Name_folder'])
data = str(args['data_folder'])

print()
print('Name model = {}'.format(Name))
print('data name = {}'.format(data))







folder_name = str(Name)
path = 'trained_flows_and_curves/'











def read_data(batch):
    path_name ="data_gwcosmo/galaxy_catalog/training_data_from_MLP/"
    data_name = "_data_250000_N_SNR_8_Nelect_10__Full_para_v1.csv".format(batch)
    GW_data = pd.read_csv(path_name+data_name,skipinitialspace=True, usecols=['H0', 'dl','m1', 'm2','a1', 'a2', 'tilt1', 
                                                                              'tilt2', 'RA', 'dec', 'theta_jn', 'phi_jl', 
                                                                             'phi_12', 'polarization', 'geo_time'])
    return GW_data

list_data = [] 
for i in range(1):
    list_data.append(read_data(i+1))


GW_data = pd.concat(list_data)
print()
print('Preview the data: SIZE = ({},{})'.format(len(GW_data), len(GW_data.columns)))
print()
print((GW_data.head()))   



data = GW_data[['dl','m1', 'm2','a1', 'a2', 'tilt1', 'tilt2', 'RA', 'dec', 'theta_jn', 'phi_jl', 
                                                                             'phi_12', 'polarization', 'geo_time', 'H0']]

#transform Polar into cartesian and spins to sigmoids
def spherical_to_cart(dl, ra, dec):
    
    x,y,z = spherical_to_cartesian(dl, dec, ra)
    return x,y,z

coordinates= data[['dl', 'RA', 'dec']]
dl = np.array(coordinates.dl)
ra = np.array(coordinates.RA)
dec = np.array(coordinates.dec)

x,y,z = spherical_to_cart(dl, ra, dec)

data['xcoord'] = x
data['ycoord'] = y
data['zcoord'] = z

# spins = data[['a1','a2']]

# a1_logit = logit(torch.from_numpy(np.array(spins.a1)))
# a2_logit = logit(torch.from_numpy(np.array(spins.a2)))

# data['a1_logit'] = a1_logit
# data['a2_logit'] = a2_logit



data = data[['xcoord', 'ycoord', 'zcoord', 'm1', 'm2','a1', 'a2', 'tilt1', 'tilt2', 'theta_jn', 'phi_jl', 
                                                                             'phi_12', 'polarization', 'geo_time', 'H0']]

def load_hyperparameters_scalers_flow(flow_name):
    torch.set_num_threads(1)
    
    #Open hyperparameter dictionary
    path = "/data/wiay/federico/PhD/cosmoflow/COSMOFlow/train_flow/trained_flows_and_curves/"+flow_name+"/"
    hyper = open(path+'hyperparameters.txt', 'r').read()
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
    scalerfile_x = path+'scaler_x.sav'
    scalerfile_y = path+'scaler_y.sav'
    scaler_x = pickle.load(open(scalerfile_x, 'rb'))
    scaler_y = pickle.load(open(scalerfile_y, 'rb'))
  

    #Open flow model file flow.pt
    flow_load = torch.load( path+'flow.pt', map_location=device)

    if flow_type == 'RealNVP':
        flow_empty = RealNVP(n_inputs= n_inputs,
            n_transforms= n_transforms,
            n_neurons= n_neurons,
            n_conditional_inputs = n_conditional_inputs,
            n_blocks_per_transform = n_blocks_per_transform,
            batch_norm_between_transforms=True,
            dropout_probability=dropout,
            linear_transform=None)
    elif flow_type == 'CouplingNSF':   
            flow_empty = CouplingNSF(n_inputs= n_inputs,
            n_transforms= n_transforms,
            n_neurons= n_neurons,
            n_conditional_inputs = n_conditional_inputs,
            n_blocks_per_transform = n_blocks_per_transform,
            batch_norm_between_transforms=True,
            dropout_probability=dropout,
            linear_transform=None)
    
    flow_empty.load_state_dict(flow_load)
    flow = flow_empty
    flow.eval()
    
    return flow, hyperparameters, scaler_x, scaler_y





print()
print('Making Probability-Probability plot with Validation data')
print()

def Flow_samples(conditional, n):
    
    Y_H0_conditional = scaler_y.transform(conditional.reshape(-1,1))

    
    conditional = np.array(Y_H0_conditional)
    data = np.array(conditional)
    data_scaled = torch.from_numpy(data.astype('float32'))
    
    flow.eval()
    flow.to('cpu')
    

    with torch.no_grad():
        samples = flow.sample(n, conditional=data_scaled.to('cpu'))
        samples= scaler_x.inverse_transform(samples.to('cpu'))
    return samples 



np.random.seed(1234)
Nresults =200
Nruns = 1
labels = ['x','y', 'z','m1', 'm2','a1', 'a2', 'tilt1','tilt2', 'theta_jn', 'phi_jl', 
                                                                             'phi_12', 'psi', 'time']
priors = {}
for jj in range(14):
    priors.update({f"{labels[jj]}": Uniform(0, 1, f"{labels[jj]}")})




for x in range(Nruns):
    results = []
    for ii in tqdm(range(Nresults)):
        posterior = dict()
        injections = dict()
        i = 0 
        for key, prior in priors.items():

            inx = np.random.randint(len(Y_scale_val))  
            truths=  scaler_x.inverse_transform(X_scale_val[inx,:].reshape(1,-1))[0]
            conditional_sample = scaler_y.inverse_transform(Y_scale_val[inx].reshape(1,-1))[0]
            conditional_sample = conditional_sample *np.ones(10000)
            samples = Flow_samples(conditional_sample, 10000)
            posterior[key] = samples[:,i] 
            injections[key] = truths[i].astype('float32').item()
            i += 1

        posterior = pd.DataFrame(dict(posterior))
        result = bilby.result.Result(
            label="test",
            injection_parameters=injections,
            posterior=posterior,
            search_parameter_keys=injections.keys(),
        priors = priors )
        results.append(result)

    fig = bilby.result.make_pp_plot(results, filename=path+folder_name+'/PP',
                              confidence_interval=(0.68, 0.90, 0.99, 0.9999))








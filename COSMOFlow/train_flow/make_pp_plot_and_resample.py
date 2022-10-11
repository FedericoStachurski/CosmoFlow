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
os.chdir('..')


flow_name = 'test_3_big_batch_log_spline'
#Open hyperparameter dictionary
path = "train_flow/trained_flows_and_curves/"+flow_name+"/"
hyper = open(path+'hyperparameters.txt', 'r').read()
hyperparameters = eval(hyper)
log_it = hyperparameters['log_it']


def read_data(batch):
    path_name ="data_gwcosmo/galaxy_catalog/training_data_from_MLP/"
    data_name = "batch_{}_250000_N_SNR_10_Nelect_10__Full_para_v2.csv".format(batch)
    GW_data = pd.read_csv(path_name+data_name,skipinitialspace=True, usecols=['H0', 'dl','m1', 'm2','a1', 'a2', 'tilt1', 
                                                                              'tilt2', 'RA', 'dec', 'theta_jn', 'phi_jl', 
                                                                             'phi_12', 'polarization', 'geo_time'])
    return GW_data

list_data = [] 
for i in range(4):
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

def cartesian_to_spher(x,y,z):
    
    dl, dec, ra= cartesian_to_spherical(x,y,z)
    return dl, dec, ra

coordinates= data[['dl', 'RA', 'dec']]
dl = np.array(coordinates.dl)
ra = np.array(coordinates.RA)
dec = np.array(coordinates.dec)

x,y,z = spherical_to_cart(dl, ra, dec)

data.loc[:,'xcoord'] = x
data.loc[:,'ycoord']= y
data.loc[:,'zcoord']= z


def logit_data(data_to_logit):
    a1_logit = logit(torch.from_numpy(np.array(data_to_logit.a1)))
    a2_logit = logit(torch.from_numpy(np.array(data_to_logit.a2)))
    phijl_logit = logit(torch.from_numpy(np.array(data_to_logit.phi_jl)))
    phi12_logit = logit(torch.from_numpy(np.array(data_to_logit.phi_12)))
    pol_logit = logit(torch.from_numpy(np.array(data_to_logit.polarization)))
    tc_logit = logit(torch.from_numpy(np.array(data_to_logit.geo_time)))

    data_to_logit.loc[:,'a1'] = np.array(a1_logit)
    data_to_logit.loc[:,'a2'] = np.array(a2_logit)
    data_to_logit.loc[:,'phi_jl'] = np.array(phijl_logit)
    data_to_logit.loc[:,'phi_12'] = np.array(phi12_logit)
    data_to_logit.loc[:,'polarization'] = np.array(pol_logit)
    data_to_logit.loc[:,'geo_time'] = np.array(tc_logit)
    return data_to_logit

def sigmoid_data(data_to_sigmoid):
    a1_sigmoid= sigmoid(torch.from_numpy(np.array(data_to_sigmoid.a1)))
    a2_sigmoid = sigmoid(torch.from_numpy(np.array(data_to_sigmoid.a2)))
    phijl_sigmoid = sigmoid(torch.from_numpy(np.array(data_to_sigmoid.phi_jl)))
    phi12_sigmoid = sigmoid(torch.from_numpy(np.array(data_to_sigmoid.phi_12)))
    pol_sigmoid = sigmoid(torch.from_numpy(np.array(data_to_sigmoid.polarization)))
    tc_sigmoid = sigmoid(torch.from_numpy(np.array(data_to_sigmoid.geo_time)))

    data_to_sigmoid.loc[:,'a1'] = np.array(a1_sigmoid)
    data_to_sigmoid.loc[:,'a2'] = np.array(a2_sigmoid)
    data_to_sigmoid.loc[:,'phi_jl'] = np.array(phijl_sigmoid)
    data_to_sigmoid.loc[:,'phi_12'] = np.array(phi12_sigmoid)
    data_to_sigmoid.loc[:,'polarization'] = np.array(pol_sigmoid)
    data_to_sigmoid.loc[:,'geo_time'] = np.array(tc_sigmoid)
    return data_to_sigmoid



# data = data[['xcoord', 'ycoord', 'zcoord', 'm1', 'm2','a1', 'a2', 'tilt1', 'tilt2', 'theta_jn', 'phi_jl', 
#                                                                              'phi_12', 'polarization', 'geo_time', 'H0']]

data = data[['dl', 'RA', 'dec', 'm1', 'm2','a1', 'a2', 'tilt1', 'tilt2', 'theta_jn', 'phi_jl', 
                                                                             'phi_12', 'polarization', 'geo_time', 'H0']]

print(data.head(10))

def scale_data(data_to_scale):
    target = data_to_scale[data_to_scale.columns[0:-1]]
    conditioners = np.array(data_to_scale[data_to_scale.columns[-1]]).reshape(-1,1)
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaled_target = scaler_x.fit_transform(target) 
    scaled_conditioners = scaler_y.fit_transform(conditioners)  
    scaled_data = np.hstack((scaled_target, scaled_conditioners))
    scaled_data = pd.DataFrame(scaled_data, index=data_to_scale.index, columns=data_to_scale.columns)
    return scaler_x, scaler_y, scaled_data
    
scaler_x, scaler_y, scaled_data = scale_data(data)
if log_it is True:
    logit_data(scaled_data)
    scaled_data = scaled_data[np.isfinite(scaled_data).all(1)]


x_train, x_val = train_test_split(scaled_data, test_size=0.25)

train_tensor = torch.from_numpy(np.asarray(x_train).astype('float32'))
val_tensor = torch.from_numpy(np.asarray(x_val).astype('float32'))


X_scale_train = train_tensor[:,:-1]
Y_scale_train = train_tensor[:,-1]
X_scale_val = val_tensor[:,:-1]
Y_scale_val = val_tensor[:,-1]







def load_hyperparameters_scalers_flow(flow_name):
    torch.set_num_threads(1)
    
    #Open hyperparameter dictionary
    path = "train_flow/trained_flows_and_curves/"+flow_name+"/"
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




flow, hyper_dict, scaler_x, scaler_y = load_hyperparameters_scalers_flow(flow_name)



def Flow_samples(conditional, n):
    
    Y_H0_conditional = scaler_y.transform(conditional.reshape(-1,1))

    
    conditional = np.array(Y_H0_conditional)
    data = np.array(conditional)
    data_scaled = torch.from_numpy(data.astype('float32'))
    
    flow.eval()
    flow.to('cpu')
    

    with torch.no_grad():
        samples = (flow.sample(N, conditional=data_scaled.to('cpu'))).detach().numpy()
        
        dict_rand = {'x':list(samples[:,0]), 'y':list(samples[:,1]), 'z':list(samples[:,2]), 
                          'm1':list(samples[:,3]), 'm2':list(samples[:,4]),'a1':list(samples[:,5]),
                           'a2':list(samples[:,6]), 'tilt1':list(samples[:,7]), 'tilt2':list(samples[:,8]),
                          'theta_jn':list(samples[:,9]), 'phi_jl':list(samples[:,10]), 'phi_12':list(samples[:,11]),
                          'polarization':list(samples[:,12]), 'geo_time':list(samples[:,13])}

        samples = pd.DataFrame(dict_rand)
        
        if log_it == 1:
            samples = sigmoid_data(samples)
        
        
        samples= scaler_x.inverse_transform(np.array(samples))
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
        inx = np.random.randint(len(Y_scale_val)) 
        i = 0 
        for key, prior in priors.items():

             
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

    fig = bilby.result.make_pp_plot(results, filename='train_flow/trained_flows_and_curves/'+flow_name+'/PP',
                              confidence_interval=(0.68, 0.90, 0.99, 0.9999))
    
N = 1000000
H0_samples = np.random.uniform(20,120,N)
    
samples = Flow_samples(H0_samples, N)
print(samples)
x = np.array(samples[:,0])
y = np.array(samples[:,1])
z = np.array(samples[:,2])

dl, dec, RA = cartesian_to_spher(x, y, z)


samples[:,0] = dl
samples[:,1] = RA
samples[:,2] = dec

dict_rand = {'dl':list(samples[:,0]), 'RA':list(samples[:,1]), 'dec':list(samples[:,2]), 
                          'm1':list(samples[:,3]), 'm2':list(samples[:,4]),'a1':list(samples[:,5]),
                           'a2':list(samples[:,6]), 'tilt1':list(samples[:,7]), 'tilt2':list(samples[:,8]),
                          'theta_jn':list(samples[:,9]), 'phi_jl':list(samples[:,10]), 'phi_12':list(samples[:,11]),
                          'polarization':list(samples[:,12]), 'geo_time':list(samples[:,13])}

samples = pd.DataFrame(dict_rand)



        




c1 = corner.corner(samples, bins = 50, plot_datapoints=False, smooth = False, levels = (0.5, 0.9), color = 'red', hist_kwargs = {'density' : 1}, hist_bin_factor=5,
                   range = [(0, 6000), (0, 2*np.pi), (-np.pi/2, np.pi/2), (5,80), (5,80), (0,0.99), (0,0.99),(0,np.pi), (0,np.pi), (0,np.pi), (0,2*np.pi),(0,2*np.pi), (0,np.pi), (0,86400)])
#data = logit_data(data)
fig = corner.corner(data[['dl', 'RA', 'dec', 
                          'm1', 'm2','a1',
                          'a2', 'tilt1', 'tilt2',
                          'theta_jn', 'phi_jl', 'phi_12',
                          'polarization', 'geo_time']] , bins = 50,
                    plot_datapoints=False, 
                    smooth = False, 
                    fig = c1, 
                    hist_bin_factor=5,
                    levels = (0.5, 0.9), 
                    plot_density=True,
                    range = [(0, 6000),  (0, 2*np.pi), (-np.pi/2, np.pi/2),
                          (5,80), (5,80), (0,0.99), (0,0.99),
                          (0,np.pi), (0,np.pi), (0,np.pi), (0,2*np.pi),
                          (0,2*np.pi), (0,np.pi), (0,86400)],
                    labels = [r'$dl[Mpc]$',r'$RA$',r'$dec$', 
                            r'$m_{1,z}$', r'$m_{2,z}$',r'$a_{1}$', 
                            r'$a_{2}$', r'$tilt_{1}$', r'$tilt_{2}$', 
                            r'$\theta_{JN}$', r'$\phi_{JL}$',  
                            r'$\phi_{12}$',r'$\psi$', r'$t_{geo}$'], 
                    hist_kwargs = {'density' : 1})



plt.savefig('train_flow/trained_flows_and_curves/'+flow_name+'/flow_resample.png', dpi = 300)    
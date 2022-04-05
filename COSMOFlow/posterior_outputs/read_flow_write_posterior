open data loaders
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
from scipy.interpolate import splrep, splev
from scipy.integrate import dblquad, quad, tplquad
from scipy.stats import norm , gaussian_kde
from tqdm import tqdm 
from scipy.stats import ncx2
import scipy as sp
from scipy import stats
import h5py
import pandas as pd
import shutil
import argparse


#pass arguments 
# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-GW_name", "--gravitational_wave", required=True,
   help="gravitational_wave to use")
ap.add_argument("-Flow_name", "--Flow_name", required=True,
   help="the flow to use to compute the posterior")



args = vars(ap.parse_args())
Name_GW = str(args['gravitational_wave'])
Name_Flow = str(args['Flow_name'])

path = "posteriors/"

print()
print('GW = {}'.format(Name_GW))
print('Flow = {}'.format(Name_Flow))
print()

#check if directory exists
if os.path.exists(path+folder_name):
    #if yes, delete directory
    shutil.rmtree(path+folder_name)

#Save model in folder
os.mkdir(path+'posterior_H0_'+Name_GW)




def load_hyperparameters_scalers_flow(flow_name):
    torch.set_num_threads(1)
    
    #Open hyperparameter dictionary
    path = "../train_flow/trained_flows_and_curves/"+flow_name+"/"
    hyper = open(path+'hyperparameters.txt', 'r').read()
    hyperparameters = eval(hyper)
    
    device = 'cpu'
    n_inputs = hyperparameters['n_inputs']
    n_conditional_inputs = hyperparameters['n_conditional_inputs'] 
    n_neurons = hyperparameters['n_neurons']
    n_transforms = hyperparameters['n_transforms']
    n_blocks_per_transform = hyperparameters['n_blocks_per_transform']
    
    #open scaler_x and scaler_y
    scalerfile_x = path+'scaler_x.sav'
    scalerfile_y = path+'scaler_y.sav'
    scaler_x = pickle.load(open(scalerfile_x, 'rb'))
    scaler_y = pickle.load(open(scalerfile_y, 'rb'))
  

    #Open flow model file flow.pt
    flow_load = torch.load( path+'flow.pt')

    flow_empty = RealNVP(n_inputs= n_inputs,
        n_transforms= n_transforms,
        n_neurons= n_neurons,
        n_conditional_inputs = n_conditional_inputs,
        n_blocks_per_transform = n_blocks_per_transform,
        batch_norm_between_transforms=True,
        dropout_probability=0.0,
        linear_transform=None)
    
    
    flow_empty.load_state_dict(flow_load)
    flow = flow_empty
    flow.eval()
    flow.to(device)
    return flow, hyperparameters, scaler_x, scaler_y




flow, hyper_dict, scaler_x, scaler_y = load_hyperparameters_scalers_flow('flow_v5_REAL')







path_name = '/data/wiay/federico/PhD/GW_data_GWTC/GWTC-1_sample_release/'

eventname = ["GW150914", "GW151012", "GW151226", "GW170104", "GW170729", "GW170814", "GW170818", "GW170823"]


index = 7

with h5py.File(path_name+eventname[index]+"_GWTC-1.hdf5", "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])
    print(eventname[index])





pd.options.mode.chained_assignment = None


GW_data = pd.DataFrame(np.array(data))
data_testing = GW_data[['luminosity_distance_Mpc','m1_detector_frame_Msun', 'm2_detector_frame_Msun', 'spin1', 'spin2','costilt1', 'costilt2','right_ascension','declination','costheta_jn']]



H0 = np.ones(len(data_testing))*70
RN = np. random.normal(0,1, size= len(data_testing))
z = np. random.uniform(low = 0.001, high = 1, size= len(data_testing))

data_testing['H0'] = H0
data_testing['RN'] = RN
data_testing['z'] = z


data_testing = data_testing[['H0', 'RN', 'luminosity_distance_Mpc','m1_detector_frame_Msun', 'm2_detector_frame_Msun', 'spin1', 'spin2','costilt1', 'costilt2','right_ascension','declination','costheta_jn', 'z']]

testing_conditionals = data_testing[['luminosity_distance_Mpc','m1_detector_frame_Msun', 'm2_detector_frame_Msun', 'spin1', 'spin2','costilt1', 'costilt2','right_ascension','declination','costheta_jn', 'z']]
testing_target = data_testing[['H0','RN']]

Y_scale_testing = scaler_y.transform(testing_conditionals[['luminosity_distance_Mpc','m1_detector_frame_Msun', 'm2_detector_frame_Msun', 'spin1', 'spin2','costilt1', 'costilt2','right_ascension','declination','costheta_jn', 'z']])



n_conditional_inputs = hyper_dict['n_conditional_inputs']

#loop over every posterior sample point Npost

Npost = 1000


like_per_point = []
points = 200
npoints = points**2

Hvec = np.linspace(20,120, 200)
dH = np.diff(Hvec[20:180])[0]








def Flow_posterior(conditional, n_points = 200, device = 'cpu'):
    n = n_points**(2) 
    
    device = device
    
    flow.eval()
    flow.to(device)
    
    with torch.no_grad():
        samples = flow.sample(n, conditional=conditional.to(device))
        samples= scaler_x.inverse_transform(samples)

        n1 = n_points
        x  = np.linspace(0,1,n_points)
        y  = np.linspace(0,1,n_points)
        dx = np.diff(x)[0] ; dy = np.diff(y)[0]
        xx, yy = np.meshgrid(x, y)
        xx = xx.reshape(-1,1)
        yy = yy.reshape(-1,1)

        xy_inp = torch.from_numpy(np.concatenate([xx,yy], axis=1)).float()
        logprobx = flow.log_prob(xy_inp, conditional=conditional.to(device))
        logprobx = logprobx.numpy() 
        logprobx = logprobx.reshape(n_points,n_points)

        
        return samples, logprobx


for i in tqdm(range(Npost)):
    


    mean_event =Y_scale_testing[i, :]    


    #Flow posterior
    conditional_testing = np.array(mean_event[:n_conditional_inputs].reshape(1,-1))
    data_scaled_testing = np.array(conditional_testing[:,:n_conditional_inputs])*np.ones((npoints, n_conditional_inputs))
    data_scaled_testing = torch.from_numpy(data_scaled_testing.astype('float32'))  

    samples, logprobx = Flow_posterior(data_scaled_testing, n_points = points)
    logprobx = logprobx.reshape(points,points)
    marginal_H0 = np.sum(np.exp(logprobx), axis=0)

    pdf_H0_smooth = marginal_H0 / np.sum(marginal_H0*dH)
    like_per_point.append(pdf_H0_smooth)
    
pdf_H0_avg = np.sum(like_per_point, axis=0)/len(like_per_point)









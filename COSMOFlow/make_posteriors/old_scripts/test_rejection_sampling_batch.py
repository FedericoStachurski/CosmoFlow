import matplotlib.pyplot as plt
import pandas as pd
import h5py
import numpy as np
import sys
sys.path.append("..")
from gw_functions import pdet_theta 
from tqdm import tqdm
from glasflow.flows import RealNVP, CouplingNSF
import torch 
import pickle 
import corner
from astropy.coordinates import spherical_to_cartesian, cartesian_to_spherical
from torch import logit, sigmoid
import os 
import multiprocessing 
import json
from scipy.spatial.distance import jensenshannon
from scipy import interpolate
from gw_functions.gw_SNR_v2 import run_bilby_sim
from scipy.stats import ncx2
import bilby
from astropy import cosmology
from train_flow.handle_flow import Handle_Flow

import json
import argparse
import time 
import matplotlib.pyplot as plt 
from gw_functions.pdet_theta import LikelihoodDenomiantor
from gw_functions.gw_event_names import GW_events
from cosmology_functions import utilities
from astropy import cosmology 


device = 'cuda:0'
flow_name = 'H0_OM0_O3_BBH_testing_real'
rth = 11
epoch = None
ndet = 3
run = 'O3'
path = '../train_flow/trained_flows_and_curves/'
flow_class = Handle_Flow(path, flow_name, device, epoch = epoch)


### Real O3 data
from tqdm import tqdm 
from gw_functions.gw_event_names import GW_events

gw_event_name_parameters = {'detectors':'HLV', 'run':'O3', 'population': 'BBH'}
gw_event_name_class = GW_events(gw_event_name_parameters)
GW_events = gw_event_name_class.get_event()
Npost_samples_event = 100

counter = 0

O3_dataGW = pd.DataFrame()
for event in tqdm(GW_events):
    GW_data = utilities.load_data_GWTC(event)
    GW_data = GW_data[['luminosity_distance', 'mass_1', 'mass_2', 'ra', 'dec']].head(Npost_samples_event)
    O3_dataGW = pd.concat((O3_dataGW, GW_data))
    counter+=1
    # if counter == 1:
    #     break
    

def get_posterior_Nevent_Nsample_Npost(prior_samples, Nsample, Nevents,  Npost, flowclass):
    N = Nsample
    Nevents = Nevents
    Npost_samples = Npost
    
    likelihood_eval = flow_class.p_theta_OMEGA_test_batching(O3_dataGW, prior_samples)
    test_like = likelihood_eval.reshape(Nevents,Npost_samples,N)
    log_likelihood = np.zeros(N)

    for i in range(N):
        like_prior = test_like[:,:,i]
        like_temp_events = np.zeros(Nevents)
        for j in range(Nevents):
            like_event = like_prior[j, :]
            like_temp_events[j] = np.log(np.sum(np.exp(like_event)))
            # like_event = np.log(np.sum(np.exp(like_event)))
            # print(np.shape(like_event))
        log_likelihood[i] = np.sum(like_temp_events)
    return log_likelihood


N_post = 100
N_samples = 100
N_events = 21
log_posterior = [] 
prior = [] 
N_thresh = 2500
counter = 0 
N_vals = []
ESS_vals = []
time_list = [] 
while True:
    start = time.time()
    prior_samples = utilities.make_samples(N_samples, dimensions = 2)
    likelihood_eval = flow_class.p_theta_OMEGA_test_batching(O3_dataGW, prior_samples) 
    log_posterior.append(get_posterior_Nevent_Nsample_Npost(prior_samples, N_samples, N_events, N_post, flow_class))
    prior.append(prior_samples)
    
    wj = np.concatenate(log_posterior)
    wj = np.exp(wj - np.max(wj))
    N = np.sum(wj)
    ESS = np.sum(wj)**(2)  /  np.sum(wj**(2))
    
    N_vals.append(N)
    ESS_vals.append(ESS)
    
    counter += 1
    efficiency = counter * N_samples / N
    
    sys.stdout.write('\rN = {} | ESS = {}  | efficiency = {}'.format(N, ESS, efficiency))
    
    end = time.time()
    time_list.append(end - start)
    plt.plot(np.cumsum(np.array(time_list)), np.array(N_vals), '-k', linewidth = 4)
    plt.xlabel(r'time [$s$]', fontsize = 13)
    plt.ylabel(r'$N = <w_{i}> $', fontsize = 13)
    plt.axhline(y = N_thresh ,color = 'r', linewidth = 3,  label = 'Ntotal')
    plt.grid(True)
    plt.yscale('log')
    plt.savefig('posterior_samples_rejection/statistics_rejection_sampling.png', dpi = 300)
    plt.show()
    plt.close()
    
    
    
    if N>= N_thresh:
        break
        
        
        

        
prior_points = np.concatenate(prior, axis = 1)
logs_temp = np.concatenate(log_posterior)#np.reshape(log_posterior, (-1, N_samples)).sum(axis=-1)#batch#np.concatenate(posterior)
weights_temp = logs_temp - np.max(logs_temp)
t_i = np.log(np.random.uniform(0,1, len(logs_temp)))
inx_accept = np.where(t_i < weights_temp)[0]
accepted_points = prior_points[:,inx_accept]
dictionary_samples = {'H0':accepted_points[0,:], 'Om0':accepted_points[1,:]}


with open('posterior_samples_rejection/_FLOW_{}_Ntheta_{}_Nomega_{}_Ntotal_{}_data.pickle'.format(flow_name, N_post, N_samples, N_thresh, run), 'wb') as handle:
        pickle.dump(dictionary_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)



import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import h5py
import numpy as np 
import pandas as pd 
from cosmology_functions import utilities 
from scipy.stats import truncnorm
from train_flow.handle_flow import Handle_Flow
import argparse
from tqdm import tqdm 
import h5py
import time 
import pickle 
import matplotlib.pyplot as plt
from gw_functions.gw_event_names import GW_events

import numpy as np
from scipy.stats import norm

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger




ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-runs", "--runs", nargs='+', required=True,
   help="make data from detector: OPTIONS [O1, O2, O3..]", default = 'O3')
ap.add_argument("-flow_name", "--flow_name", required=True,
   help="name_of_flow")
ap.add_argument("-Ntheta", "--Ntheta", required=True,
   help="Number of GW posterior points to be used in batching the flow", default = 100)
ap.add_argument("-device", "--device", required=True,
   help="cpu, cuda:0 ...", default = 'cuda:0')




args = vars(ap.parse_args())
runs = str(args['runs'])
flow_name = str(args['flow_name'])
N_theta = int(args['Ntheta'])
device = str(args['device'])


output = "posterior_samples_rejection/nested_sampling_folders/run_{}_FLOW_{}_Ntheta_{}_data".format(runs,flow_name,N_theta)
logger = setup_logger(output=output)
# os.chdir("..")
path = '../train_flow/trained_flows_and_curves/'

def get_data_GW(events, N_theta):
    data_total = pd.DataFrame()
    for event in tqdm(events): 
        df = utilities.load_data_GWTC(event)
        df = df[['luminosity_distance', 'mass_1', 'mass_2', 'ra', 'dec']].head(N_theta)
        data = [data_total, df]
        data_total = pd.concat(data)
    return data_total


if 'O1' in runs:
    flow_name_o1 = flow_name+'_O1'
    flow_class_o1 = Handle_Flow(path, flow_name_o1, device, epoch = None)
    data_o1 = get_data_GW(GW_class.get_event('O1', 'HL'), N_theta)
    n_conditional = flow_class_o1.hyperparameters['n_conditional_inputs']
    
if 'O2' in runs:
    flow_name_o2_hlv = flow_name+'_O2_HLV'
    flow_class_o2_hlv = Handle_Flow(path, flow_name_o2_hlv, device, epoch = None)
    data_o2_hlv = get_data_GW(GW_class.get_event('O2', 'HLV'), N_theta)
    n_conditional = flow_class_o2_hlv.hyperparameters['n_conditional_inputs'] 

    flow_name_o2_hl = flow_name+'_O2_HL'
    flow_class_o2_hl = Handle_Flow(path, flow_name_o2_hl, device, epoch = None)
    data_o2_hl = get_data_GW(GW_class.get_event('O2', 'HL'), N_theta)
    
    
if 'O3' in runs: 
    flow_name_o3_hlv = flow_name#+'_O3_HLV'
    flow_class_o3_hlv = Handle_Flow(path, flow_name_o3_hlv, device, epoch = None)
    gw_event_name_parameters = {'detectors':'HLV', 'run':'O3', 'population': 'BBH'}
    gw_event_name_class = GW_events(gw_event_name_parameters)
    GW_events = gw_event_name_class.get_event()
    Nevents_O3 = len(GW_events)
    data_o3_hlv = get_data_GW(GW_events, N_theta)
    n_conditional = flow_class_o3_hlv.hyperparameters['n_conditional_inputs']


    
    # flow_name_o3_hl = flow_name+'_O3_HL'
    # flow_class_o3_hl = Handle_Flow(path, flow_name_o3_hl, device, epoch = None)
    # data_o3_hl = get_data_GW(GW_class.get_event('O3', 'HL'), N_theta)
    
    
def posterior_eval_flow(data, prior_samples, flow_class_variable, N_events, N_theta, N_samples):
    likelihood_eval = flow_class_variable.p_theta_H0_one_go_batch(data, prior_samples.T)
    likelihood_eval = np.reshape(likelihood_eval, (int(N_events*N_theta),int(N_samples)))
    posterior_eval = flow_class_variable.get_posterior_from_batch(likelihood_eval, N_samples,  N_theta , N_events)
    return posterior_eval

# dictionary_samples = {'H0':accepted_points[0,:], 'gamma':accepted_points[1,:], 'kappa':accepted_points[2,:], 'zp':accepted_points[3,:],
#                      'alpha':accepted_points[4,:], 'beta':accepted_points[5,:], 'mmax':accepted_points[6,:], 'mmin':accepted_points[7,:],
#                      'mu_g':accepted_points[8,:], 'sigma_g':accepted_points[9,:], 'lambda_peak':accepted_points[10,:], 'delta_m':accepted_points[11,:],


#     h0_samples_proposal, _ = proposal_pdf(20,180,N_samples)
#     gamma_samples_proposal, _ = proposal_pdf(0,12,N_samples)
#     kappa_samples_proposal, _ = proposal_pdf(0,6,N_samples)
#     zp_samples_proposal, _ = proposal_pdf(0,4,N_samples)
    
#     alpha_samples_proposal, _ = proposal_pdf(1.5,12,N_samples)
#     beta_samples_proposal, _ = proposal_pdf(-4.0,12,N_samples)
#     mmax_samples_proposal, _ = proposal_pdf(50.0,200.0,N_samples)
#     mmin_samples_proposal, _ = proposal_pdf(2.0,10.0,N_samples)
    
#     mug_samples_proposal, _ = proposal_pdf(20.0,50.0,N_samples)
#     sigmag_samples_proposal, _ = proposal_pdf(0.4,10.0,N_samples)
#     lambda_samples_proposal, _ = proposal_pdf(0.0,1.0,N_samples)
#     delta_samples_proposal, _ = proposal_pdf(0.0,10.0,N_samples)

class p_Omega(Model):
    """Prior space"""
    def __init__(self):
        # Names of parameters to sample
        # self.names = ["H0", "gamma", "kappa", "zp", "alpha", "beta", "mmax", "mmin", "mu_g", "sigma_g", "lambda_peak", "delta_m"]
        
        # Prior bounds for each parameter
        # self.bounds = {"H0":[20,180], "gamma":[0,12], "kappa":[0,6],
        #                "zp":[0,4], "alpha":[1.5,12], "beta":[-4,12],
        #                "mmax":[50,200], "mmin":[2,10], "mu_g":[20,50], "sigma_g":[0.4,10],
        #                "lambda_peak":[0,1], "delta_m":[0,10]}
        self.names = ["H0", "Om0"]
        self.bounds = {"H0":[20,140], "Om0":[0,1]}

    def log_prior(self, x):
        """
        Returns log of prior given a live point assuming uniform
        priors on each parameter.
        """
        # Check if values are in bounds, returns True/False
        # Then take the log to get 0/-inf and make sure the dtype is float
        log_p = np.log(self.in_bounds(x), dtype="float")
        # Iterate through each parameter (x and y)
        # since the live points are a structured array we can
        # get each value using just the name
        for n in self.names:
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        return log_p

    def log_likelihood(self, x):
        """
        Returns log likelihood of given live point
        """
        prior_samples = self.unstructured_view(x).T
        # print(np.ndim(prior_samples))
        log_prob = np.zeros(1)
        # print(prior_samples)
        if np.ndim(prior_samples) == 1:
            prior_samples = np.reshape(prior_samples, (flow_class_o3_hlv.hyperparameters['n_conditional_inputs'],1))
        if 'O1' in runs:
            posterior_o1 = posterior_eval_flow(data_o1, prior_samples, flow_class_o1, len(GW_class.get_event('O1', 'HL')), N_theta, 1)
            log_prob += posterior_o1
        if 'O2' in runs:
            log_posterior_o2_hlv = posterior_eval_flow(data_o2_hlv, prior_samples, flow_class_o2_hlv, len(GW_class.get_event('O2', 'HLV')), N_theta, 1)
            log_posterior_o2_hl = posterior_eval_flow(data_o2_hl, prior_samples, flow_class_o2_hl, len(GW_class.get_event('O2', 'HL')), N_theta, 1)
            log_prob += log_posterior_o2_hlv + log_posterior_o2_hlv
        if 'O3' in runs:    
            log_posterior_o3_hlv = posterior_eval_flow(data_o3_hlv, prior_samples, flow_class_o3_hlv, Nevents_O3, N_theta, 1)
            # log_posterior_o3_hl = posterior_eval_flow(data_o3_hl, prior_samples, flow_class_o3_hl, len(GW_class.get_event('O3', 'HL')), N_theta, 1)
            # log_prob += log_posterior_o3_hlv + log_posterior_o3_hl
            log_prob = log_posterior_o3_hlv

        return log_prob

    
    
fs = FlowSampler(p_Omega(), output=output, resume=False, seed=1234)

# And go!
fs.run()
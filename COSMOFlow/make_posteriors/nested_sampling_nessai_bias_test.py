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
import torch
import numpy as np
from scipy.stats import norm
import multiprocessing as mp
# mp.set_start_method('spawn')
# torch.multiprocessing.set_start_method('spawn')
import torch.multiprocessing
from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger
from nessai.utils.multiprocessing import initialise_pool_variables



ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-name_folder", "--name_folder", required=True,
   help="Folder name", default = 'Test')
ap.add_argument("-runs", "--runs", nargs='+', required=True,
   help="make data from runs: OPTIONS [O1,O2,O3,...]", default = 'O3')
ap.add_argument("-det", "--det", nargs='+', required=True,
   help="make data from detector: OPTIONS [H1,L1,V1]", default = ['H1','L1','V1'])
ap.add_argument("-flow_name", "--flow_name", required=True,
   help="name_of_flow")
ap.add_argument("-batches", "--batches", required=True,
   help="Number of batches of the testing data", default = 1)
ap.add_argument("-device", "--device", required=True,
   help="cpu, cuda:0 ...", default = 'cuda:0')




args = vars(ap.parse_args())
name = str(args['name_folder'])
runs = args['runs']
det = args['det']
flow_name = str(args['flow_name'])
device = str(args['device'])

output = "posterior_samples_rejection/nested_sampling_folders/{}_run_{}_FLOW_{}_data".format(name,runs[0],flow_name)
logger = setup_logger(output=output)
# os.chdir("..")
path = '../train_flow/trained_flows_and_curves/'
#run_O3_det_['H1', 'L1', 'V1']_name_testing_catalog_True_band_K_batch_1_N_1000_SNR_11_Nelect_50__Full_para_v1
#run_O3_det_['H1', 'L1', 'V1']_name_testing_catalog_True_band_K_batch_1_N_1000_SNR_11_Nelect_50__Full_para_v1.csv

def read_data(batch):
    path_name ="../data_cosmoflow/galaxy_catalog/testing_data_from_MLP/"
    data_name = "run_O3_det_['H1', 'L1', 'V1']_name_BBH_data_full_sky_NSIDE_32_OM0_catalog_True_band_K_batch_1_N_200_SNR_11_Nelect_2__Full_para_v1.csv".format(batch)
    GW_data = pd.read_csv(path_name+data_name,skipinitialspace=True)
                     #      usecols=['snr', 'H0','gamma','kappa','zp', 'alpha', 'beta', 'mmax', 'mmin', 'mu_g', 'sigma_g', 'lambda_peak', 'delta_m',
                     # 'luminosity_distance', 'mass_1', 'mass_2', 'ra', 'dec',
                     # 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'theta_jn',
                     # 'phi_jl', 'phi_12', 'psi','geocent_time', 'app_mag', 'inx'])
    return GW_data


list_data = [] 
for i in range(1):
    list_data.append(read_data(i+1))

dataGW = pd.concat(list_data)
dataGW = dataGW.drop_duplicates(keep='first').sample(frac=1)    
dataGW = dataGW[['luminosity_distance', 'mass_1', 'mass_2', 'ra', 'dec']]
N_events = len(dataGW)
    

    
    
    
def posterior_eval_flow(data, prior_samples, flow_class_variable, N_events, N_theta, N_samples):
    likelihood_eval = flow_class_variable.p_theta_H0_one_go_batch(data, prior_samples.T)
    likelihood_eval = np.reshape(likelihood_eval, (int(N_events*N_theta),int(N_samples)))
    posterior_eval = flow_class_variable.get_posterior_from_batch(likelihood_eval, N_samples,  N_theta , N_events)
    return posterior_eval


    
if 'O3' in runs: 
    flow_name_o3_hlv = flow_name
    flow_class_o3_hlv = Handle_Flow(path, flow_name_o3_hlv, device, epoch = None)
    data_o3_hlv = dataGW
    n_conditional = flow_class_o3_hlv.hyperparameters['n_conditional_inputs']
    

class p_Omega(Model):
    """Prior space"""
    def __init__(self):
        
        # Names of parameters to sample
        # self.names = ["H0", "gamma", "kappa", "zp", "alpha", "beta", "mmax", "mmin", "mu_g", "sigma_g", "lambda_peak", "delta_m"]
        # self.names = ["H0", "gamma", "mmax", "mu_g"]
        self.names = ["H0", "Om0"]
        # Prior bounds for each parameter
        # self.bounds = {"H0":[20,180], "gamma":[0,12], "kappa":[0,6],
        #                "zp":[0,4], "alpha":[1.5,12], "beta":[-4,12],
        #                "mmax":[50,200], "mmin":[2,10], "mu_g":[20,50], "sigma_g":[0.4,10],
        #                "lambda_peak":[0,1], "delta_m":[0,10]}
        # self.bounds = {"H0":[20,180], "gamma":[0,12],
        #         "mmax":[50,200], "mu_g":[20,50]}
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
            prior_samples = np.reshape(prior_samples, (2,1))

        if 'O3' in runs:    
            log_posterior_o3_hlv = posterior_eval_flow(data_o3_hlv, prior_samples, flow_class_o3_hlv, N_events, 1, 1)
            # log_posterior_o3_hl = posterior_eval_flow(data_o3_hl, prior_samples, flow_class_o3_hl, len(GW_class.get_event('O3', 'HL')), N_theta, 1)
            log_prob += log_posterior_o3_hlv

        return log_prob

if __name__ == "__main__":


    mp = torch.multiprocessing.get_context("spawn")
    pool = mp.Pool(
        processes=1,
        initializer=initialise_pool_variables,
        initargs=(p_Omega(),),
    )
    
    # mp.set_start_method('spawn')
    fs = FlowSampler(p_Omega(), output=output, resume=False, seed=1234, n_pool = 1, pool = pool)

    # And go!
    fs.run()
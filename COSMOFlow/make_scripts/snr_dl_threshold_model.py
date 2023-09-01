import os, sys
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import numpy as np 
import pandas as pd
import pickle
import time
from scipy.interpolate import splrep, splev
from scipy.integrate import dblquad, quad, tplquad
from tqdm import tqdm
import matplotlib.pyplot as plt
import bilby 
import astropy.constants as const
from MultiLayerPerceptron.validate import run_on_dataset
from MultiLayerPerceptron.nn.model_creation import load_mlp
import argparse
from poplar.nn.networks import LinearModel, load_model
from poplar.nn.rescaling import ZScoreRescaler
import torch
from cosmology_functions import cosmology

from gw_functions import gw_priors_v2

from tqdm import tqdm 
import multiprocessing
from scipy.stats import loguniform



import argparse

import matplotlib.pyplot as plt 
from gw_functions.pdet_theta import LikelihoodDenomiantor
from astropy import cosmology 





#pass arguments 
# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-Model", "--Name_model", required=True,
   help="Name of the folder to save the GW_posteriors")
ap.add_argument("-Nsamples", "--samples", required=True,
   help="samples to use")
ap.add_argument("-ndet", "--ndet", required=True,
   help="number of detectors used")
ap.add_argument("-quantile", "--quantile", required=True,
   help="quantile threshold")




args = vars(ap.parse_args())
Model = str(args['Name_model'])
Nsamples= int(args['samples'])
quantile= float(args['quantile'])
device = 'cuda:0'



path = 'models/MLP_models/'
np.random.seed(1122)

N = Nsamples
# print(os.getcwd())
# model_v1 = load_model(path_models+'models/new_model/v1_SNR_v1O3/model.pth', device = device)
model = load_model(path+'{}/model.pth'.format(Model), device = device)
# zmax_list = [] 
snr_max_list = [] 
distance_list = []
snr_dl_list = [] 
for i in tqdm(range(100)):
    snr = []
    #sample GW priors

    distributions = {'mass': 'Uniform'}
    _, _, _, a1sample, a2sample, tilt1sample, tilt2sample,RAsample, decsample, theta_jnsample, phi_jlsample, phi_12sample, psisample, phasesample, geo_time = gw_priors_v2.draw_prior(N)


    dlsample = loguniform.rvs(100, 11_000, size=N) # np.random.uniform(10,11_000,N)
    m1zsample = np.ones(N)*112*(1+2) # loguniform.rvs(2, 350, size=N) # np.random.uniform(2,350,N)
    m2zsample = np.ones(N)*112*(1+2)# loguniform.rvs(2, 350, size=N) # np.random.uniform(2,350,N)

    inx = np.where(m1zsample < m2zsample)[0]
    temp_m1 = m1zsample[inx]
    temp_m2 = m2zsample[inx]
    m1zsample[inx] = temp_m2
    m2zsample[inx] = temp_m1


    data = { 'luminosity_distance':dlsample, 'mass_1':m1zsample, 'mass_2':m2zsample,'a_1': a1sample, 'a_2': a2sample, 'tilt_1': tilt1sample, 'tilt_2': tilt2sample,
                 'ra':RAsample, 'dec':decsample,'theta_jn':theta_jnsample, 'phi_jl': phi_jlsample, 'phi_12': phi_12sample,
            'psi':psisample, 'phase': 0, 'geocent_time': geo_time}

    def prep_data_for_MLP(df):
        data_testing = df[['mass_1','mass_2','theta_jn', 'ra', 'dec','psi', 'geocent_time','a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_jl', 'phi_12' ]]
        df[['geocent_time']] = df[['geocent_time']]%86164.0905
        data_testing = df[['mass_1','mass_2','theta_jn', 'ra', 'dec','psi', 'geocent_time','a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_jl', 'phi_12' ]]
        xdata_testing = torch.as_tensor(data_testing.to_numpy(), device=device).float()
        return xdata_testing

    GW_data = pd.DataFrame(data)

    ### SNR calulcator using MLPs ###make_scripts
    x_data_MLP  = prep_data_for_MLP(GW_data)
    ypred = model.run_on_dataset(x_data_MLP.to(device))
    snr_pred = ypred.cpu().numpy()
    network_snr_sq = np.sum(snr_pred[:,]**2, axis = 1)
    network_snr = np.sqrt(network_snr_sq)
    # GW_data['snr*dl'] = network_snr
    snr_dl_list.append(network_snr)
    
    
quantile_Value_snr_dl = np.quantile(np.array(snr_dl_list).flatten(), quantile)
np.savetxt(path+'{}/snr_dl_th.txt'.format(Model), str(quantile_Value_snr_dl), delimiter=',')

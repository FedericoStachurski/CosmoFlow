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

import matplotlib.pyplot as plt 
from gw_functions.pdet_theta import LikelihoodDenomiantor
from gw_functions.gw_event_names import GW_events
from cosmology_functions import utilities
from astropy import cosmology 

#pass arguments 
# # Construct the argument parser
# ap = argparse.ArgumentParser()

# # Add the arguments to the parser
# ap.add_argument("-Folder", "--Name_folder", required=True,
#    help="Name of the folder to save the GW_posteriors")
# ap.add_argument("-Nsamples", "--samples", required=True,
#    help="Posterior samples to use")
# ap.add_argument("-SNRth", "--SNRth", required=True,
#    help="SNR threshold")
# ap.add_argument("-Flow", "--Flow", required=True,
#    help="Trained flow to use")
# ap.add_argument("-population", "--population", required=True,
#    help="Trained flow to use")
# ap.add_argument("-run", "--run", required=True,
#    help="Trained flow to use")
# ap.add_argument("-det", "--detectors", required=True,
#    help="detectors used [e.g. HLV, HL ... ]")
# ap.add_argument("-device", "--device", required=True, default = 'cpu',
#    help="device to use: cpu or cuda")
# ap.add_argument("-epoch", "--epoch", required=False, default = None,
#    help="which epoch nubmer to use")




# args = vars(ap.parse_args())
# Folder = str(args['Name_folder'])
# Nsamples= int(args['samples'])
# rth= int(args['SNRth'])
# Flow= str(args['Flow'])
# population= str(args['population'])
# run = str(args['run'])
# detectors= str(args['detectors'])
# device= str(args['device'])
# ndet = len(detectors)
# print('Detectors used: '+str(ndet))
# epoch = args['epoch']
# if epoch is not None:
#     epoch= int(args['epoch'])
# h0 = np.linspace(20,140,500)


gw_event_name_parameters = {'detectors':'HLV', 'run':'O3', 'population': 'BBH'}
gw_event_name_class = GW_events(gw_event_name_parameters)
GW_events = gw_event_name_class.get_event()
print('Performing analysis on these events: {}'.format(GW_events))
GW_data = utilities.load_data_GWTC(GW_events[0])

cosmo_bilby = cosmology.FlatLambdaCDM(H0 = 70, Om0 = 0.3)
denominator_class = LikelihoodDenomiantor(rth, cosmo_bilby, ndet)
path = '../train_flow/trained_flows_and_curves/'
flow_name = Flow
flow_class = Handle_Flow(path, flow_name, device, epoch = epoch)

check if directory exists
if os.path.exists(Folder) is False:
    #Save model in folder
    os.mkdir(Folder)
    os.mkdir(Folder+'/plots')
    os.mkdir(Folder+'/posteriors')
    os.mkdir(Folder+'/posteriors_no_w')
    os.mkdir(Folder+'/O3_H0_post')
    os.mkdir(Folder+'/JS_means')





def get_likelihoods(h0,om0, df, N_samples, flow_class):
    likelihood_space = []
    for h in tqdm(h0):
        likelihood_vertical.append(flow_class.p_theta_H0_full_single(df.loc[:N_samples], h))
    return  np.array(likelihood_vertical).T
    
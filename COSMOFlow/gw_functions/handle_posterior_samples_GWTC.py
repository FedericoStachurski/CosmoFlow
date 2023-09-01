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



class Handle_GWTC(object):
    def __init__(self, path, flow_hyper, snrth, ndet):
        self.path = path 
        self.flow_hyper



def load_data_GWTC(event, xyz = 0 ):
    if int(event[2:8]) <= 190930:
        
        path_gw = '/data/wiay/federico/PhD/GWTC_2.1/'
        file_name = path_gw+'IGWN-GWTC2p1-v2-{}_PEDataRelease_mixed_nocosmo.h5'.format(event)
    else:   
        path_gw = '/data/wiay/federico/PhD/GWTC_3/'
        file_name = path_gw+'IGWN-GWTC3p0-v1-{}_PEDataRelease_mixed_nocosmo.h5'.format(event)
    
    d = h5py.File(file_name,'r')
    samples = np.array(d.get('C01:IMRPhenomXPHM/posterior_samples'))
    d.close()
    df = pd.DataFrame(samples)
    return df
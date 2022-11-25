import os, sys
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
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
print(parentdir)

from gw_functions import gw_priors
from gw_functions import gw_SNR
from tqdm import tqdm 
import multiprocessing

#type_data = 'training'
type_data = 'testing'


N = 2000
snr = []
#sample GW priors

#dl, m1z, m2z, a1, a2, tilt1, tilt2, RA, dec, theta_jn, psi =  [], [] , [] ,[], [] , [], [] ,[] ,[] ,[], []

distributions = {'mass': 'Uniform'}
_, _, _, a1sample, a2sample, tilt1sample, tilt2sample,RAsample, decsample, theta_jnsample, phi_jlsample, phi_12sample, psisample, _, geo_time = gw_priors.draw_prior(N,distributions)

dlsample = np.random.uniform(10,15_000,N)
m1zsample = np.random.uniform(1,150,N)
m2zsample = np.random.uniform(1,150,N)


para_array = np.vstack((dlsample, m1zsample, m2zsample, a1sample, a2sample, tilt1sample, tilt2sample,RAsample, decsample, theta_jnsample, phi_jlsample, phi_12sample, psisample, geo_time)).T

def snr_func(parameters):
    dlsample, m1zsample, m2zsample, a1sample, a2sample, tilt1sample, tilt2sample, RAsample, decsample, theta_jnsample,phi_jlsample, phi_12sample, psisample, geo_time= parameters
    return gw_SNR.SNR_from_inj( dlsample, m1zsample, m2zsample, a1sample, a2sample, tilt1sample, tilt2sample, RAsample, decsample, theta_jnsample, phi_jlsample, phi_12sample, psisample, 0, geo_time)


with multiprocessing.Pool(5) as p:
    snr = list(tqdm(p.imap(snr_func,para_array), total = N))




dlsample, m1zsample, m2zsample, a1sample, a2sample, tilt1sample, tilt2sample,RAsample, decsample, theta_jnsample, _, _, psisample
if type_data == 'training':
    data = { 'dl':dlsample, 'm1z':m1zsample, 'm2z':m2zsample,'a1': a1sample, 'a2': a2sample, 'tilt1': tilt1sample, 'tilt2': tilt2sample,
             'RA':RAsample, 'dec':decsample,'thteta_jn':theta_jnsample, 'phi_jl': phi_jlsample, 'phi_12': phi_12sample, 'polarization':psisample, 'geo_time': geo_time, 'snr':snr}
    df = pd.DataFrame(data)
    print(df)

    path_data = r"data_for_MLP/data_sky_theta/training/"

    df.to_csv(path_data+'_data_{}_full_para_v2_batch_1.csv'.format(N))
    
if type_data == 'testing':
    data = { 'dl':dlsample, 'm1z':m1zsample, 'm2z':m2zsample,'a1': a1sample, 'a2': a2sample, 'tilt1': tilt1sample, 'tilt2': tilt2sample,
             'RA':RAsample, 'dec':decsample,'thteta_jn':theta_jnsample, 'phi_jl': phi_jlsample, 'phi_12': phi_12sample, 'polarization':psisample, 'geo_time': geo_time, 'snr':snr}
    df = pd.DataFrame(data)
    print(df)

    path_data = r"data_for_MLP/data_sky_theta/testing/"

    df.to_csv(path_data+'_data_{}_full_para_v2_batch_1.csv'.format(N))
    
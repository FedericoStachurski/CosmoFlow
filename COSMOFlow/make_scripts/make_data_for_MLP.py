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

type_data = 'training'
#type_data = 'testing'


N = 100_000


snr = []
#sample GW priors

dl, m1z, m2z, a1, a2, tilt1, tilt2, RA, dec, theta_jn, phase =  [], [] , [] ,[], [] , [], [] ,[] ,[] ,[], []
for i in tqdm(range(N)):


    dlsample, m1zsample, m2zsample, a1sample, a2sample, tilt1sample, tilt2sample,RAsample, decsample, theta_jnsample, _, _, _, phase_sample = gw_priors.draw_prior(1)
#     dlsample = np.random.uniform(10,24000, 1)[0]

    snr.append(gw_SNR.SNR_from_inj( dlsample[0], m1zsample, m2zsample, a1sample[0], a2sample[0], tilt1sample[0], tilt2sample[0], RAsample[0], decsample[0], theta_jnsample[0], 0, 0, 0, phase_sample))
    
    dl.append(dlsample[0])
    m1z.append(m1zsample)
    m2z.append(m2zsample)
    a1.append(a1sample[0]) 
    a2.append(a2sample[0]) 
    tilt1.append(tilt1sample[0]) 
    tilt2.append(tilt2sample[0]) 
    RA.append(RAsample[0]) 
    dec.append(decsample[0]) 
    theta_jn.append(theta_jnsample[0])
#     phi_jl.append(phi_jlsample)
#     phi_12.append(phi_12sample)
#     psi.append(psisample)
    phase.append(phase_sample[0])


if type_data == 'training':
    data = { 'dl':dl, 'm1z':m1z, 'm2z':m2z,'a1': a1, 'a2': a2, 'tilt1': tilt1, 'tilt2': tilt2,
             'RA':RA, 'dec':dec,'thteta_jn':theta_jn, 'phase':phase, 'snr':snr}
    df = pd.DataFrame(data)
    print(df)

    path_data = r"data_for_MLP/data_sky_theta/training/"

    df.to_csv(path_data+'_data_{}_sky_theta_phase_v1.csv'.format(N))
    
elif type_data == 'testing':  
    data = { 'dl':dl, 'm1z':m1z, 'm2z':m2z,'a1': a1, 'a2': a2, 'tilt1': tilt1, 'tilt2': tilt2,
             'RA':RA, 'dec':dec,'thteta_jn':theta_jn,'phase':phase, 'snr':snr}
    df = pd.DataFrame(data)
    print(df)

    path_data = r"data_for_MLP/data_sky_theta/testing/"

    df.to_csv(path_data+'_data_{}_sky_theta_v2.csv'.format(N))
    
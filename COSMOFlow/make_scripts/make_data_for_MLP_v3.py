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
from gw_functions import gw_SNR_v2
from tqdm import tqdm 
import multiprocessing
from scipy.stats import loguniform


#np.random.seed(122456)
#np.random.seed(12211)



import argparse



#pass arguments 
#Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-Name", "--Name_file", required=True,
   help="Name of data")
ap.add_argument("-type", "--type_data", required=True,
   help="type of data: OPTIONS[training, testing]")
ap.add_argument("-seed", "--seed", required=False,
   help="seed of the data", default = 1996)
ap.add_argument("-detector", "--detector", nargs='+', required=True,
   help="make data from detector: OPTIONS [H1, L1, V1]", default = 'H1')
ap.add_argument("-run", "--run", required=True,
   help="Observing run: OPTIONS [O1, O2, O3, O4] ", default = 'O3')
ap.add_argument("-N", "--N", required=True,
   help="n samples in the data set", default = 100_000)
ap.add_argument("-threads", "--threads", required=False,
   help="threads", default = 10)
ap.add_argument("-approximator", "--wave_approx", required=True,
   help="wave approximator", default = 'IMRPhenomPv2')
ap.add_argument("-batch", "--batch", required=True,
   help="threads", default = 1)


args = vars(ap.parse_args())
Name = str(args['Name_file'])
type_of_data = str(args['type_data'])
N = int(args['N'])
threads = int(args['threads'])
seed = int(args['seed'])
det = args['detector']
run = str(args['run'])
threads = int(args['threads'])
batch = int(args['batch'])
approximator = str(args['wave_approx'])


print(det)
np.random.seed(seed)



#type_data = 'training'
type_data = type_of_data


N = N
snr = []
#sample GW priors

distributions = {'mass': 'Uniform'}
_, _, _, a1sample, a2sample, tilt1sample, tilt2sample,RAsample, decsample, theta_jnsample, phi_jlsample, phi_12sample, psisample, phasesample, geo_time = gw_priors.draw_prior(N,distributions)


dlsample = loguniform.rvs(100, 11_000, size=N) # np.random.uniform(10,11_000,N)
m1zsample = loguniform.rvs(2, 350, size=N) # np.random.uniform(2,350,N)
m2zsample = loguniform.rvs(2, 350, size=N) # np.random.uniform(2,350,N)

inx = np.where(m1zsample < m2zsample)[0]
temp_m1 = m1zsample[inx]
temp_m2 = m2zsample[inx]
m1zsample[inx] = temp_m2
m2zsample[inx] = temp_m1



data = { 'luminosity_distance':dlsample, 'mass_1':m1zsample, 'mass_2':m2zsample,'a_1': 0, 'a_2': 0, 'tilt_1': 0, 'tilt_2': 0,
             'ra':RAsample, 'dec':decsample,'theta_jn':theta_jnsample, 'phi_jl': 0, 'phi_12': 0, 'psi':psisample, 'phase': 0, 'geocent_time': geo_time}

df = pd.DataFrame(data)

print(df.loc[500:520])
def compute_SNR(inx):
    return gw_SNR_v2.run_bilby_sim(df, inx, det, run, approximator)


threads = threads
indicies = np.arange(N)
SNRs_list = []



with multiprocessing.Pool(threads) as p:
    SNRs = list(tqdm(p.imap(compute_SNR,indicies), total = N))
SNRs = np.array(SNRs).T


if type_data == 'training':
    
    data = { 'luminosity_distance':dlsample, 'mass_1':m1zsample, 'mass_2':m2zsample, 'ra':RAsample, 'dec':decsample, 'theta_jn':theta_jnsample,
            'psi':psisample,'geocent_time': geo_time}
    
    if det == ['H1', 'L1', 'V1']:
        data.update({'snr_H1':SNRs[0], 'snr_L1':SNRs[1], 'snr_V1':SNRs[2], 'snr_network':SNRs[3]}) 
        df = pd.DataFrame(data)
        print(df)

        path_data = r"data_for_MLP/data_sky_theta/training/"
        df.to_csv(path_data+'_{}_{}_det_{}_{}_{}_run_{}_approx_{}_batch_{}.csv'.format(Name, N, *det, run, approximator, batch))
        
    elif det == ['H1', 'L1']:
        data.update({'snr_H1':SNRs[0], 'snr_L1':SNRs[1], 'snr_network':SNRs[2]}) 
        df = pd.DataFrame(data)
        print(df)

        path_data = r"data_for_MLP/data_sky_theta/training/"
        df.to_csv(path_data+'_{}_{}_det_{}_{}_run_{}_approx_{}_batch_{}.csv'.format(Name, N, *det, run, approximator, batch))



    
if type_data == 'testing':
    data = { 'luminosity_distance':dlsample, 'mass_1':m1zsample, 'mass_2':m2zsample, 'ra':RAsample, 'dec':decsample, 'theta_jn':theta_jnsample,
            'psi':psisample,'geocent_time': geo_time}
    
    if det == ['H1', 'L1', 'V1']:
        data.update({'snr_H1':SNRs[0], 'snr_L1':SNRs[1], 'snr_V1':SNRs[2], 'snr_network':SNRs[3]}) 
        df = pd.DataFrame(data)
        print(df)
        path_data = r"data_for_MLP/data_sky_theta/testing/"
        df.to_csv(path_data+'testing_data_{}_det_{}_{}_{}_run_{}_batch_{}.csv'.format(N, *det, run, batch))
        
    elif det == ['H1', 'L1']:
        data.update({'snr_H1':SNRs[0], 'snr_L1':SNRs[1], 'snr_network':SNRs[2]}) 
        df = pd.DataFrame(data)
        print(df)
        path_data = r"data_for_MLP/data_sky_theta/testing/"
        df.to_csv(path_data+'testing_data_{}_det_{}_{}_run_{}_batch_{}.csv'.format(N, *det, run, batch))


    
    

    
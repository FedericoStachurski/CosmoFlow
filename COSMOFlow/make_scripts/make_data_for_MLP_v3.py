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

from gw_functions import gw_priors_v2
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
   help="wave approximator", default = 'IMRPhenomXPHM')
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

# distributions = {'mass': 'Uniform'}
_, _, _, a1sample, a2sample, tilt1sample, tilt2sample,RAsample, decsample, theta_jnsample, phi_jlsample, phi_12sample, psisample, phasesample, geo_time = gw_priors_v2.draw_prior(N)


dlsample = np.random.uniform(10,11_000,N) #loguniform.rvs(100, 11_000, size=N) # 
m1zsample = np.random.uniform(4,350,N) #loguniform.rvs(4, 350, size=N) # 
# m2zsample = np.random.uniform(4,350,N) #loguniform.rvs(4, 350, size=N) # BBH m2 
m2zsample = np.random.uniform(0.5,15,N) #loguniform.rvs(4, 350, size=N) # NSBH m2 
#COMMENT! 
#Thin kabout maybe using p(m1,m2) from gwcosmo

inx = np.where(m1zsample < m2zsample)[0]
temp_m1 = m1zsample[inx]
temp_m2 = m2zsample[inx]
m1zsample[inx] = temp_m2
m2zsample[inx] = temp_m1

data = { 'luminosity_distance':dlsample, 'mass_1':m1zsample, 'mass_2':m2zsample,'a_1': a1sample, 'a_2': a2sample, 'tilt_1': tilt1sample, 'tilt_2': tilt2sample,
             'ra':RAsample, 'dec':decsample,'theta_jn':theta_jnsample, 'phi_jl': phi_jlsample, 'phi_12': phi_12sample, 'psi':psisample, 'phase': 0, 'geocent_time': geo_time} #COMMENT!


df = pd.DataFrame(data)

# print(df.loc[500:520])
def compute_SNR(inx):
    return gw_SNR_v2.run_bilby_sim(df, inx, det, run, approximator, snr_type = 'optimal')


threads = threads
indicies = np.arange(N)
SNRs_list = []


with multiprocessing.Pool(threads) as p:
    SNRs = list(tqdm(p.imap(compute_SNR,indicies), total = N))
SNRs = np.array(SNRs).T


    
# data = { 'luminosity_distance':dlsample, 'mass_1':m1zsample, 'mass_2':m2zsample, 'ra':RAsample, 'dec':decsample, 'theta_jn':theta_jnsample,
#         'psi':psisample,'geocent_time': geo_time}
data = { 'luminosity_distance':dlsample, 'mass_1':m1zsample, 'mass_2':m2zsample,'a_1': a1sample, 'a_2': a2sample, 'tilt_1': tilt1sample, 'tilt_2': tilt2sample,
             'ra':RAsample, 'dec':decsample,'theta_jn':theta_jnsample, 'phi_jl': phi_jlsample, 'phi_12': phi_12sample, 'psi':psisample, 'geocent_time': geo_time}

data.update({'snr_H1':SNRs[0], 'snr_L1':SNRs[1], 'snr_V1':SNRs[2]}) 
df = pd.DataFrame(data)


if type_data == 'training':
    path_data = r"data_for_MLP/training/"
    if len(det) == 3:
        df.to_csv(path_data+'_{}_{}_det_{}_{}_{}_run_{}_approx_{}_batch_{}.csv'.format(Name, N, *det, run, approximator, batch))
    elif len(det) == 2:
        df.to_csv(path_data+'_{}_{}_det_{}_{}_run_{}_approx_{}_batch_{}.csv'.format(Name, N, *det, run, approximator, batch))
elif type_data == 'testing':
    path_data = r"data_for_MLP/testing/"
    if len(det) == 3:
        df.to_csv(path_data+'testing_data_{}_det_{}_{}_{}_run_{}_batch_{}.csv'.format(N, *det, run, batch))
    elif len(det) == 2: 
        df.to_csv(path_data+'testing_data_{}_det_{}_{}_run_{}_batch_{}.csv'.format(N, *det, run, batch))
        



    

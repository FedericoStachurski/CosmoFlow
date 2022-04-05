# Author: Federico Stachurski 
#Procedure: generate data frame of GW observables 
#Input: -N number of evetns, -batch_size how many batches to save from N, -SNRth threshold, -zmax redhsift maximum -mth magnitude threshold 
#Output: data frame of galaxies
#Date: 07/08/2021



#%% import libraries
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
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
from cosmology_functions import priors 
from cosmology_functions import cosmology 
from scipy.stats import chi2
import numpy.random as rn
from scipy.stats import ncx2
from scipy.interpolate import splrep, splev
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import argparse

#pass arguments 
# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-N", "--Nevents", required=True,
   help="number of events", default=10000)
ap.add_argument("-batches", "--batches", required=True,
   help="number of batches", default=10)
ap.add_argument("-type", "--type", required=True,
   help="training or testing data?", default = 'training')
ap.add_argument("-SNRth", "--threshold", required=True,
   help="SNR threshold of GW event selection", default = 8)
ap.add_argument("-zmax", "--maximum_redshift", required=True,
   help="maximum redshift to sample from", default = 1)
ap.add_argument("-mth", "--magnitude_threshold", required=True,
   help="magnitude threshold", default = 24)
ap.add_argument("-H0", "--H0", required=False,
   help="magnitude threshold", default = None)
ap.add_argument("-Omega_m", "--Omega_m", required=False,
   help="magnitude threshold", default = None)
ap.add_argument("-Nselect", "--Nselect", required=False,
   help="galaxies to select per observation run", default = 100)
ap.add_argument("-Nbatch", "--Nbatch", required=False,
   help="batch number starting", default = 0)
ap.add_argument("-prior_inx", "--prior_inx", required=False,
   help="prior exponent index", default = 0)


args = vars(ap.parse_args())
N = int(args['Nevents'])
type_of_data = str(args['type'])
rho_th = float(args['threshold'])
mth = float(args['magnitude_threshold'])
zmax = float(args['maximum_redshift'])
H0 = (args['H0'])
Omega_m = (args['Omega_m'])
batches = int(args['batches'])
Nselect = int(args['Nselect'])
Nbatch = int(args['Nbatch'])
prior_inx = int(args['prior_inx'])

print('EVENTS TO COLLECT = {}'.format(N))
print('BATCHES = {}'.format(batches))
print('Nselect = {}'.format(Nselect))
print('TYPE = '+ type_of_data)
print('SNRth = {}'.format(rho_th))
print('mth = {}'.format(mth))
print('Zmax = {}'.format(zmax))
print('Omega_k = 0 ')

if Omega_m is not None:
    print('Omega_m = {}'.format(Omega_m))
else:    
    print('Omega_m ~ U[0,1]')
if H0 is not None:
    print('H0 = {}'.format(H0))
else: 
    print('H0 ~ U[20,120]')
                      
                      

catalogue_name = "/data/wiay/federico/PhD/gwcosmoFLow_v2/TAO_catalogues/mock_catalogue_1.txt"

# read catalog
with open(catalogue_name, 'rb') as fp:   # Unpickling
    catalog = pickle.load(fp)


z_cat  = catalog[0]
m_cat  = catalog[1]
RA_cat = catalog[2]
dec_cat  = catalog[3]    

Ncat = len(z_cat)

print('Catalogue read correctly')
print('Ncat = '+str(Ncat))




if type_of_data == 'training':
    #path_data = r"data/new_training_data_v2/training_data_2_batches"
    path_data = r"/data/wiay/federico/PhD/gwcosmoFLow_v2/data_gwcosmo/TAO_cat/training_data/"
    
    
if type_of_data == 'testing': 
    path_data = r"/data/wiay/federico/PhD/gwcosmoFLow_v2/data_gwcosmo/TAO_cat/testing_data/"

    
#%% GW event selection
def GW_events(Nevent):
    "GW event selection for analysis of Nevents"
    
    start = time.time()
    #store parameters in empty list
    prior_mass = bilby.core.prior.Uniform(5, 100, "chirp_mass") 
    rho_list, Mz_list, z_list, H0_list, Omega_m_list, m_app_list, RA_list, dec_list = [], [], [], [], [], [], [], []
    pr_all = (1/(1+z_cat))
    rth = rho_th
    #inx_cat = set(inx[0])
    
    for i in range(Nevent):
        
        #Sample H0 
        if H0 is not None:
            H0_sample  = float(H0) 
            
        else: 
            H0_sample =  np.random.uniform(20,120)
            #H0_sample =  np.random.normal(70, 15,size = 1)[0]
            
        if Omega_m is not None:
            Omega_m_sample  = float(Omega_m)
        else: 
            Omega_m_sample = np.random.uniform(0,1)
        

        while True: 
            
            rand_inx = np.random.randint(0,Ncat, Nselect)
            zsamples = z_cat[rand_inx]
            RAsamples = RA_cat[rand_inx]
            decsamples = dec_cat[rand_inx]


            dlsamples = cosmology.z_to_dl_H_Omegas(zsamples,H0_sample, Omega_m_sample, 1 - Omega_m_sample)
            chirp_sample = prior_mass.sample(len(zsamples))
            Mz_sample = cosmology.M_z(chirp_sample,zsamples)
            
            #Compute SNR 
            rho_sample = cosmology.snr_Mz_z(Mz_sample, zsamples, H0_sample) 
            rho_obs = np.sqrt((ncx2.rvs(2, rho_sample**2, size=len(zsamples))))
            inx_obs = np.where(rho_obs > rth)[0]
            
           
  
            
            if np.array(inx_obs).size != 0:
            
                #observed
                z_all = zsamples[inx_obs]
                rho_all  = rho_obs[inx_obs]
                Mz_all = Mz_sample[inx_obs]
                RA_all = RAsamples[inx_obs]
                dec_all = decsamples[inx_obs]
                
                
                pr = 1/(1+z_all)
                draw_z = np.random.choice(z_all, 1, p = pr / np.sum(pr)) 
                z_sample = draw_z.item()
                inx_z = (np.where(z_all == z_sample)[0])[0]
                
                rho_GW = rho_all[inx_z]
                z_GW = z_all[inx_z]
                Mz_GW = Mz_all[inx_z]
                RA_GW = RA_all[inx_z]
                dec_GW = dec_all[inx_z]
                
                
                H0_list.append(H0_sample)
                rho_list.append(rho_GW)
                z_list.append(z_GW)
                Mz_list.append(Mz_GW)
                RA_list.append(RA_GW)
                dec_list.append(dec_GW)
                
                break 
        if i%1000 == 0:        
            print('Events {}/{}, SNR = {}, Mz = {}, z = {}, H0 = {}, Omega_m = {}'.format(i+1,
                                                                                          Nevent,
                                                                                          round(rho_GW,3),
                                                                                          round(Mz_GW,3), 
                                                                                          round(z_GW,3),     
                                                                                          round(H0_sample,3),   
                                                                                          round(Omega_m_sample,3)))
    end = time.time()
    passed_time = round(end - start,3)
    print('EVENTS WERE GENERATED IN {} s ({} m , {} h)'.format(passed_time, round(passed_time/60,3), round(passed_time/3600,3)))
    return H0_list, rho_list, Mz_list, z_list, RA_list, dec_list
       
          
            
os.chdir('..')
os.chdir('..')



for j in range(batches):
    if type_of_data == 'training':
        #store events
        print('BATCH = {}'.format(j+1))
        event = GW_events(int(N/batches))
        output_df = pd.DataFrame({'H0': event[0], 
                                  'SNR': event[1], 'Mz': event[2], 
                                  'z': event[3]})
        output_df.to_csv(path_data+'data_SNRth_{}_v2_test.dat'.format(int(rho_th)))
        j += 1
    if type_of_data == 'testing':
        #store events
        print('BATCH = {}'.format(j+1))
        event = GW_events(int(N/batches))
        output_df = pd.DataFrame({'H0': event[0], 
                                  'SNR': event[1], 'Mz': event[2], 
                                  'z': event[3]})
        output_df.to_csv(path_data+'data_mth_{}_SNRth_{}_H0_70_Omega_0.3_1000.dat'.format(int(mth), int(rho_th), int(N/batches)))
        j += 1
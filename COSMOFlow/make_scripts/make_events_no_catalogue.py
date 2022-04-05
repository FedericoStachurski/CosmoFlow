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
ap.add_argument("-mth", "--magnitude_threshold", required=False,
   help="magnitude threshold", default = 24)
ap.add_argument("-H0", "--H0", required=False,
   help="magnitude threshold", default = None)
ap.add_argument("-Omega_m", "--Omega_m", required=False,
   help="magnitude threshold", default = None)
ap.add_argument("-Nselect", "--Nselect", required=False,
   help="galaxies to select per observation run", default = None)
ap.add_argument("-Nbatch", "--Nbatch", required=False,
   help="batch number starting", default = 0)


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
   

    # def draw_RA_Dec(N):
#     #sample RA
#     ra_obs = np.random.uniform(0,180,N)
#     ra_obs = ra_obs* np.pi / 180
#     #sample declinations
#     P = np.random.uniform(0,1,N)
#     dec = np.arcsin(2*P-1) 
#     return ra_obs, dec



if type_of_data == 'training':
    #path_data = r"data/new_training_data_v2/training_data_2_batches"
    path_data = r"/data/wiay/federico/PhD/gwcosmoFLow_v2/data_gwcosmo/empty_catalogue/training_data"
    
    
if type_of_data == 'testing': 
    path_data = r"/data/wiay/federico/PhD/gwcosmoFLow_v2/data_gwcosmo/empty_catalogue/testing_data"

#%% GW event selection
def GW_events(Nevent):
    "GW event selection for analysis of Nevents"
    
    start = time.time()
    #store parameters in empty list
    prior_mass = bilby.core.prior.Uniform(15, 100, "chirp_mass") 
    rho_list, Mz_list, z_list, H0_list, Omega_m_list, m_app_list, RA_list, dec_list = [], [], [], [], [], [], [], []
    rth = rho_th
    #inx_cat = set(inx[0])
    
    i = 0
    while i < Nevent:
        
        #Sample H0 
        if H0 is not None:
            H0_sample  = float(H0) 
            
        else: 
            H0_sample =  np.random.uniform(20,120)
            
        if Omega_m is not None:
            Omega_m_sample  = float(Omega_m)
        else: 
            Omega_m_sample = np.random.uniform(0,1)
        

        while True: 

            #M_grid =  np.linspace(-23,-5,500)
            z_grid = np.linspace(0,zmax,100)
            pz = priors.p_z(z_grid, Omega_m_sample)

            #intepolate z vs pz
            spl_z = splrep(z_grid, pz )
            z_grid = np.linspace(0.001, zmax, 1000)
            pz = splev(z_grid, spl_z)

            #pM = priors.p_M(M_grid, H0_sample)


            #check that out of catalogue galaxy is above mth 

            #Msamples = np.random.choice(M_grid, N_out, p = pM / np.sum(pM))
            zsamples = np.random.choice(z_grid, Nselect, p = pz / np.sum(pz))


            dlsamples = cosmology.z_to_dl_H_Omegas(zsamples,H0_sample, Omega_m_sample, 1 - Omega_m_sample)
            #appmsamples = cosmology.app_mag(Msamples, dlsamples)
            chirp_sample = prior_mass.sample(len(zsamples))
            Mz_sample = cosmology.M_z(chirp_sample,zsamples)
            
            
            #Compute SNR 
            rho_sample = cosmology.snr_Mz_z(Mz_sample, zsamples, H0_sample) 
            rho_obs = np.sqrt((ncx2.rvs(2, rho_sample**2, size=len(zsamples))))
            inx_obs = np.where(rho_obs > rth)[0]
            
            #print(len(inx_obs))
            if np.array(inx_obs).size != 0:
                
                #observed
                z_sample = zsamples[inx_obs]
                rho_obs  = rho_obs[inx_obs]
                Mz_sample = Mz_sample[inx_obs]
                
                z_all = np.array(z_sample)
                pr_all = 1 / (1+z_all)
                draw_z = np.random.choice(z_all, 1, p = pr_all / np.sum(pr_all)) 
                z_obs = draw_z.item()
                inx_z = (np.where(z_all == z_obs)[0])[0]
                

                rho_obs  = rho_obs[inx_z]
                Mz_sample = Mz_sample[inx_z]

                #parameters_out = [rho_obs, z_sample , Mz_sample, app_m, RA_GW, dec_GW]
                rho_list.append(rho_obs)
                z_list.append(z_obs)
                Mz_list.append(Mz_sample)
                H0_list.append(H0_sample)
                Omega_m_list.append(Omega_m_sample)



                if i%1000 == 0:
                    print('Events {}/{}, SNR = {}, Mz = {}, z = {} , H0 = {}, Omega_m = {}'.format(i+1,
                                                                                                          Nevent,
                                                                                                          round(rho_obs,3),
                                                                                                          round(Mz_sample,3), 
                                                                                                          round(z_obs,3), 
                                                                                                          round(H0_sample,3),   
                                                                                                          round(Omega_m_sample,3)))
                    print()
                i += 1
                break
            else: 
                continue

    end = time.time()
    passed_time = round(end - start,3)
    print('EVENTS WERE GENERATED IN {} s ({} m , {} h)'.format(passed_time, round(passed_time/60,3), round(passed_time/3600,3)))
    return H0_list, Omega_m_list, rho_list, Mz_list, z_list


batches = batches

os.chdir('..')
os.chdir('..')


for j in range(batches):
    if type_of_data == 'training':
        #store events
        print('BATCH = {}'.format(j+1))
        event = GW_events(int(N/batches))
        output_df = pd.DataFrame({'H0': event[0], 'Omega_m' : event[1], 
                                  'SNR': event[2], 'Mz': event[3], 
                                  'z': event[4]  })
        output_df.to_csv(path_data+'/data_zmax_{}_SNRth_{}_constant_Omega_v6.dat'.format(zmax, int(rho_th)))
        j += 1
    if type_of_data == 'testing':
        #store events
        print('BATCH = {}'.format(j+1))
        event = GW_events(int(N/batches))
        output_df = pd.DataFrame({'H0': event[0], 'Omega_m' : event[1], 
                                  'SNR': event[2], 'Mz': event[3], 
                                  'z': event[4] })
        output_df.to_csv(path_data+'/data_zmax_{}_SNRth_{}_H0_70_Omega_0.3_1000.dat'.format(zmax, int(rho_th), int(N)))
        j += 1    
        
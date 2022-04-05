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
import random
np.random.seed(1)

#pass arguments 
# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-N", "--Nevents", required=True,
   help="number of events", default=10000)
ap.add_argument("-batches", "--batches", required=True,
   help="number of batches", default=1)
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



prior_mass = bilby.core.prior.Uniform(15, 100, "chirp_mass") 
rth = 8

zmax = 1.0
mth = 23.0
catalogue_name = "/data/wiay/federico/PhD/gwcosmoFlow_v3/catalogues/catalogue_zmax_{}_mth_{}.dat".format(zmax, mth)

# read catalog
catalog = pd.read_csv(catalogue_name,skipinitialspace=True, usecols=['Apparent_m', 'z', 'RA', 'dec'])
print('Catalogue read correctly')

#length catalogue
z_cat = catalog.z
m_cat = catalog.Apparent_m
RA_cat = catalog.RA
dec_cat = catalog.dec
Ncat = len(catalog)

#grid of z and M
M_grid =  np.linspace(-23,-5,100)
z_grid = np.linspace(0,zmax,100)


def draw_RA_Dec(N):
    #sample RA 
    ra_obs = np.random.uniform(0,180,N)
    ra_obs = ra_obs* np.pi / 180
    #sample declinations
    P = np.random.uniform(0,1,N)
    dec = np.arcsin(2*P-1) 
    return ra_obs, dec


def evolution(z):
    lam = -1
    return (1+z)**(lam)

def draw_cumulative_z(N, distribution):
    #grid = np.linspace(0.00001,zmax,100)
    cdf = np.zeros(len(z_grid))
    for i in range(len(z_grid )):
        cdf[i] = quad(lambda z: distribution(z), z_grid[0], z_grid[i])[0]
    cdf = cdf/np.max(cdf)  
    t = rn.random(N)
    samples = np.interp(t,cdf,z_grid)
    return samples

def draw_cumulative_M(N,H0, distribution):
    #grid = np.linspace(-23,-5,100)
    cdf = np.zeros(len(M_grid))
    for i in range(len(M_grid)):
        cdf[i] = quad(lambda M: distribution(M, H0),  M_grid [0], M_grid [i])[0]
    cdf = cdf/np.max(cdf)     
    t = rn.random(N)
    samples = np.interp(t,cdf,M_grid)
    return samples


pz = evolution(z_grid)  * priors.p_z(z_grid, 0.3)

#intepolate z vs pz
spl_z = splrep(z_grid, pz )

def pz_int(z):
    return splev(z, spl_z )    



pr_all = evolution(z_cat)
pr_all /= sum(pr_all)

if type_of_data == 'training':
    path_data = r"/data/wiay/federico/PhD/gwcosmoFlow_v3/data_gwcosmo/events_mock_cat_zmax_"+str(int(zmax))+"_mth_"+str(int(mth))+"_sky/training_data/"
    

if type_of_data == 'testing': 
    path_data = r"/data/wiay/federico/PhD/gwcosmoFlow_v3/data_gwcosmo/events_mock_cat_zmax_1_mth_23_sky/testing_data/"

#%% GW event selection
def GW_events(Nevent):
    "GW event selection for analysis of Nevents"
    
    start = time.time()
    #store parameters in empty list
    prior_mass = bilby.core.prior.Uniform(15, 100, "chirp_mass") 
    rho_list, Mz_list, z_list, H0_list, Omega_m_list, m_app_list, RA_list, dec_list = [], [], [], [], [], [], [], []
    rth = rho_th

    
    i = 0
    while i < Nevent:

        #Sample H0 
        if H0 is not None:
            H0_sample  = float(H0) 
        else: 
            H0_sample = np.random.uniform(20,120, 1)[0]

        #Sample omega_m
        if Omega_m is not None:
            Omega_m_sample  = float(Omega_m)
        else: 
            Omega_m_sample = np.random.uniform(0,1)
        

        while True: 
            
            #compute schecter function for given H0 sample
            pM = priors.p_M(M_grid, H0_sample)

            #sample galaxies from unvierse
            Msamples = draw_cumulative_M(Nselect, H0_sample, priors.p_M)
            zsamples = draw_cumulative_z(Nselect, pz_int)
            RA_samples, dec_samples = draw_RA_Dec(Nselect)
            
            #compute distance and apparent magnitudes
            dlsamples = cosmology.fast_z_to_dl_v2(zsamples,H0_sample)
            appmsamples = cosmology.app_mag(Msamples, dlsamples)
            
            #check which are in catalogue and which are out
            inx_out = np.where(appmsamples > mth)[0]
            inx_in = np.where(appmsamples < mth)[0]


            N_in = len(inx_in)

            if N_in != 0:
                #sample z_in and m_in redshifts from catalogue
                index_galaxy = np.random.choice(Ncat, p = pr_all , size = N_in)
                ra_in = RA_cat[index_galaxy]
                dec_in = dec_cat[index_galaxy]
                z_in = z_cat[index_galaxy]
                m_in = m_cat[index_galaxy]  
            else: 
                ra_in = []
                dec_in = []
                z_in = []
                m_in = []    

            z_out = zsamples[inx_out] ; m_out = appmsamples[inx_out]
            ra_out = RA_samples[inx_out] ; dec_out = dec_samples[inx_out]
            
            z_tot = np.concatenate([z_in, z_out], axis=0)
            m_tot = np.concatenate([m_in, m_out], axis=0)
            ra_tot = np.concatenate([ra_in, ra_out], axis=0)
            dec_tot = np.concatenate([dec_in, dec_out], axis=0)


            if np.size(z_tot) != 1: 
                random_indx = np.random.permutation(Nselect)
                z_tot = z_tot[random_indx]
                m_tot = m_tot[random_indx]
                ra_tot = ra_tot[random_indx]
                dec_tot = dec_tot[random_indx]
            
            #Dl and Mz
            chirp_sample = prior_mass.sample(Nselect)
            Mz_sample = cosmology.M_z(chirp_sample,z_tot)

            #Compute SNR 
            rho_sample = cosmology.snr_Mz_z(Mz_sample, z_tot, H0_sample) 
            rho_obs = np.sqrt((ncx2.rvs(2, rho_sample**2, size=Nselect, loc = 0, scale = 1)))


            inx_obs = np.where(rho_obs > rth)[0]
            if len(inx_obs): 
                
                #get obs galaxy
                z_samples = z_tot[inx_obs]
                m_samples = m_tot[inx_obs]
                ra_samples = ra_tot[inx_obs]
                dec_samples = dec_tot[inx_obs]
                
                #get obs GW
                rho_obs = rho_obs[inx_obs]
                Mz_sample = Mz_sample[inx_obs]
                 
            
                #store all parameters 
                RA_list.append(ra_samples[0])
                dec_list.append(dec_samples[0])
                rho_list.append(rho_obs[0])
                Mz_list.append(Mz_sample[0]) 
                z_list.append(z_samples[0]) 
                m_app_list.append(m_samples[0]) 
                H0_list.append(H0_sample)


                i += 1 
                break
                
        if i % 1 == 0:
            sys.stdout.write('\r{}/{}'.format(i, int(Nevent)))
            
            
        else: 
            continue

    end = time.time()
    passed_time = round(end - start,3)
    print()
    print('EVENTS WERE GENERATED IN {} s ({} m , {} h)'.format(passed_time, round(passed_time/60,3),round(passed_time/3600,3)))
    return H0_list, rho_list, Mz_list, z_list, m_app_list, RA_list, dec_list


batches = batches

os.chdir('..')
os.chdir('..')


for j in range(batches):
    if type_of_data == 'training':
        #store events
        print('BATCH = {}'.format(j+1))
        event = GW_events(int(N/batches))
        output_df = pd.DataFrame({'H0': event[0], 
                                  'SNR': event[1], 'Mz': event[2], 
                                  'z': event[3], 'm_app': event[4], 'RA':event[5], 'dec':event[6]
                                   })
        output_df.to_csv(path_data+'data_mth_{}_SNRth_{}_{}_batch_{}.csv'.format(int(mth), int(rho_th), int(N/batches),j+Nbatch))
        j += 1
    if type_of_data == 'testing':
        #store events
        print('BATCH = {}'.format(j+1))
        event = GW_events(int(N/batches))
        output_df = pd.DataFrame({'H0': event[0], 
                                  'SNR': event[1], 'Mz': event[2], 
                                  'z': event[3], 'm_app': event[4], 'RA':event[5], 'dec':event[6]
                                   })
        output_df.to_csv(path_data+'data_mth_{}_SNRth_{}_H0_70_Omega_0.3_1000.csv'.format(int(mth), int(rho_th), int(N/batches)))
        j += 1    
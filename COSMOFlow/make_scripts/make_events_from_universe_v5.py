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



catalogue_name = "/data/wiay/federico/PhD/gwcosmoFLow_v2/catalogues/catalogue_zmax_{}_mth_{}.dat".format(zmax, mth)

# read catalog
catalog = pd.read_csv(catalogue_name,skipinitialspace=True, usecols=['Apparent_m', 'z', 'RA', 'dec'])
print('Catalogue read correctly')


z_cat  = catalog.z
m_cat  = catalog.Apparent_m
RA_cat = catalog.RA
dec_cat  = catalog.dec

#length catalogue
Ncat = len(z_cat)

#scale phi
scale = 1.0 / (1e2)

print('Ncat = '+str(Ncat))

#length catalogue
Ncat = len(z_cat)     

def draw_RA_Dec(N):
    #sample RA 
    ra_obs = np.random.uniform(0,180,N)
    ra_obs = ra_obs* np.pi / 180
    #sample declinations
    P = np.random.uniform(0,1,N)
    dec = np.arcsin(2*P-1) 
    return ra_obs, dec

def H0U(H0):
    return H0**(0)

def H03(H0):
    return H0**(3)

def H06(H0):
    return H0**(6)

def draw_H0(N, prior):
    grid = np.linspace(20,120,1000)
    cdf = np.zeros(len(grid))
    for i in range(len(grid )):
        cdf[i] = quad(lambda H0: prior(H0), 20, grid[i])[0]
    cdf = cdf/np.max(cdf)     
    t = rn.random(N)
    samples = np.interp(t,cdf,grid)
    return samples


#Turn catalog in arrays
z_cat = np.array(z_cat)
m_cat = np.array(m_cat)
RA_cat = np.array(RA_cat)
dec_cat = np.array(dec_cat)


if type_of_data == 'training':
    #path_data = r"data/new_training_data_v2/training_data_2_batches"
    path_data = r"/data/wiay/federico/PhD/gwcosmoFLow_v2/data_gwcosmo/in&out_events_mock_cat_zmax_"+str(int(zmax))+"_mth_"+str(int(mth))+"/training_data/"
    
    
if type_of_data == 'testing': 
    path_data = r"/data/wiay/federico/PhD/gwcosmoFLow_v2/data_gwcosmo/in&out_events_mock_cat_zmax_"+str(int(zmax))+"_mth_"+str(int(mth))+"/testing_data/"

#%% GW event selection
def GW_events(Nevent):
    "GW event selection for analysis of Nevents"
    
    start = time.time()
    #store parameters in empty list
    prior_mass = bilby.core.prior.Uniform(15, 100, "chirp_mass") 
    rho_list, Mz_list, z_list, H0_list, Omega_m_list, m_app_list, RA_list, dec_list = [], [], [], [], [], [], [], []
    pr = (1/(1+z_cat))
    rth = rho_th

    
    i = 0
    while i < Nevent:
        
        #Sample H0 
        if H0 is not None:
            H0_sample  = float(H0) 
            
        else: 
            #H0_sample =  np.random.uniform(20,120)
            H0_sample = draw_H0(1, H0U)[0]
            
        if Omega_m is not None:
            Omega_m_sample  = float(Omega_m)
        else: 
            Omega_m_sample = np.random.uniform(0,1)
        

        rho_temp, Mz_temp, z_temp, RA_temp, dec_temp, m_app_temp = [], [], [], [], [], []


        while True: 
            
            Vcomoving = cosmology.Vc(zmax, H0_sample, 0.3)
            
            #Define number density 
            n = 0.002 * (H0_sample/50)**(3)
            scaled_n = n * scale
            Ngalaxies = int(cosmology.Ngal(Vcomoving, scaled_n))
            inx_galaxies = np.random.randint(0,Ngalaxies-1,Nselect)
            inx_cat = inx_galaxies[np.where(inx_galaxies < len(z_cat))[0]]
            N_in = len(inx_cat)
            
            if N_in: 
                N_out = len(inx_galaxies) - N_in
            else:
                N_out = len(inx_galaxies)


            
            if N_out < 1:
                continue
            
            M_grid =  np.linspace(-23,-5,100)
            z_grid = np.linspace(0,zmax,100)
            pz = priors.p_z(z_grid, Omega_m_sample)

            #intepolate z vs pz
            spl_z = splrep(z_grid, pz )
            z_grid = np.linspace(0.001, zmax, 1000)
            pz = splev(z_grid, spl_z)

            pM = priors.p_M(M_grid, H0_sample)
            temp, temp_m = [], []
            
            while True: 
                #check that out of catalogue galaxy is above mth 
                Msamples = np.random.choice(M_grid, N_out, p = pM / np.sum(pM))
                zsamples = np.random.choice(z_grid, N_out, p = pz / np.sum(pz))

                dlsamples = cosmology.z_to_dl_H_Omegas(zsamples,H0_sample, Omega_m_sample, 1 - Omega_m_sample)
                appmsamples = cosmology.app_mag(Msamples, dlsamples)
                inx_out = np.where(appmsamples > mth)[0]
                temp.append(zsamples[inx_out])
                temp_m.append(appmsamples[inx_out])
                z_out = np.concatenate(temp, axis=0)
                m_out = np.concatenate(temp_m, axis=0)

                if len(z_out) >= N_out:
                    z_out = z_out[:N_out]
                    m_out = m_out[:N_out]
                    break 

            if N_in:
                index_galaxy = np.random.randint(0, len(z_cat), N_in)
                z_in = z_cat[index_galaxy]
                m_in  = m_cat[index_galaxy]
                z_samples = np.concatenate((z_out, z_in), axis=0)
                m_samples = np.concatenate((m_out, m_in), axis=0)
            else: 
                z_samples = z_out
                m_samples = m_out

            temp_indicies = np.linspace(0, int(len(z_samples) - 1), len(z_samples), dtype='int')
            random.shuffle(temp_indicies)
            z_samples = z_samples[np.array(temp_indicies)]
            m_samples = m_samples[np.array(temp_indicies)]


            
            
            #Dl and Mz
            chirp_sample = prior_mass.sample(len(z_samples))
            Mz_sample = cosmology.M_z(chirp_sample,z_samples)

            #Compute SNR 
            rho_sample = cosmology.snr_Mz_z(Mz_sample, z_samples, H0_sample) 
            rho_obs = [] 
            for rho in rho_sample:
                rho_obs.append(np.sqrt((ncx2.rvs(2, rho**2, size=1)))[0])


            #rho_obs = np.sqrt((ncx2.rvs(2, rho_sample**2, size=len(rho_sample))))

            #print(rho_obs)
            inx_obs = np.where(np.array(rho_obs) > rth)[0]
            
            if len(inx_obs): 
                z_samples = z_samples[inx_obs]
                m_samples = m_samples[inx_obs]
                rho_samples = np.array(rho_obs)[inx_obs]
                Mz_samples = Mz_sample[inx_obs]
                z_all = z_samples
                pr_all = 1 / (1+z_all)
                draw_z = np.random.choice(z_all, 1, p = pr_all / np.sum(pr_all)) 
                z_sample = draw_z.item()
                inx_z = (np.where(z_all == z_sample)[0])[0]

                
                rho_list.append(rho_samples[inx_z])
                Mz_list.append(Mz_samples[inx_z]) 
                z_list.append(z_sample) 
                m_app_list.append(m_samples[inx_z]) 
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
    return H0_list, rho_list, Mz_list, z_list, m_app_list


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
                                  'z': event[3], 'm_app': event[4],
                                   })
        output_df.to_csv(path_data+'data_mth_{}_SNRth_{}_{}_batch_{}.dat'.format(int(mth), int(rho_th), int(N/batches),j+Nbatch))
        j += 1
    if type_of_data == 'testing':
        #store events
        print('BATCH = {}'.format(j+1))
        event = GW_events(int(N/batches))
        output_df = pd.DataFrame({'H0': event[0], 
                                  'SNR': event[1], 'Mz': event[2], 
                                  'z': event[3], 'm_app': event[4],
                                   })
        output_df.to_csv(path_data+'data_mth_{}_SNRth_{}_H0_70_Omega_0.3_1000.dat'.format(int(mth), int(rho_th), int(N/batches)))
        j += 1    
        
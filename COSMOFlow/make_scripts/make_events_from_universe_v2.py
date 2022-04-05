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


#path = "/data/wiay/federico/PhD/gwcosmoFLow/data/universes/"
#path_data_training = "/data/wiay/federico/PhD/gwcosmoFLow/data/"
catalogue_name = "/data/wiay/federico/PhD/gwcosmoFLow/gwcosmoFlow_v2/catalogues/catalogue_zmax_{}_mth_{}.dat".format(zmax, mth)

# read catalog
catalog = pd.read_csv(catalogue_name,skipinitialspace=True, usecols=['Apparent_m', 'z', 'RA', 'dec'])
print('Catalogue read correctly')
#print(os.getcwd())

z_cat  = catalog.z
m_cat  = catalog.Apparent_m
RA_cat = catalog.RA
dec_cat  = catalog.dec

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


#Turn catalog in arrays
z_cat = np.array(z_cat)
m_cat = np.array(m_cat)
RA_cat = np.array(RA_cat)
dec_cat = np.array(dec_cat)

print('Ncat = {}'.format(Ncat))


if type_of_data == 'training':
    #path_data = r"data/new_training_data_v2/training_data_2_batches"
    path_data = r"data/new_training_data_v2/constant_omega_training_data"
    
    
if type_of_data == 'testing': 
    path_data = r"data/new_training_data_v2/testing"

#%% GW event selection
def GW_events(Nevent):
    "GW event selection for analysis of Nevents"
    
    start = time.time()
    #store parameters in empty list
    prior_mass = bilby.core.prior.Uniform(15, 100, "chirp_mass") 
    rho_list, Mz_list, z_list, H0_list, Omega_m_list, m_app_list, RA_list, dec_list = [], [], [], [], [], [], [], []
    pr = (1/(1+z_cat))
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
        


        rho_temp, Mz_temp, z_temp, RA_temp, dec_temp, m_app_temp = [], [], [], [], [], []

        while True: 

            Vcomoving = cosmology.Vc(zmax, H0_sample, Omega_m_sample)
            #Define number density 
            n = 1.61 * 100 / (H0_sample/100)**(3)
            scaled_n = n * (1/1e6)
            Ngalaxies = cosmology.Ngal(Vcomoving, scaled_n)
            inx_galaxies = np.random.randint(0,Ngalaxies-1,Nselect)
            inx_cat = inx_galaxies[np.where(inx_galaxies < len(z_cat))[0]]
            N_in = len(inx_cat)

            if N_in: 
                N_out = len(inx_galaxies) - N_in
            else:
                N_out = len(inx_galaxies)



            if N_out < 2:
                continue
            M_grid =  np.linspace(-23,-5,500)
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

            #print('N_out > mth = {}'.format(len(z_out)))
            chirp_sample = prior_mass.sample(len(z_out))
            Mz_sample = cosmology.M_z(chirp_sample,z_out)
            #Compute SNR 
            rho_sample = cosmology.snr_Mz_z(Mz_sample, z_out, H0_sample) 
            rho_obs = np.sqrt((ncx2.rvs(2, rho_sample**2, size=len(z_out))))
            inx_obs = np.where(rho_obs > rth)[0]
            #print(len(inx_obs))
            if np.array(inx_obs).size != 0:
                #observed
                z_sample = z_out[inx_obs]
                rho_obs  = rho_obs[inx_obs]
                Mz_sample = Mz_sample[inx_obs]
                app_m = m_out[inx_obs]
                RA_GW, dec_GW = draw_RA_Dec(N_out)
                #parameters_out = [rho_obs, z_sample , Mz_sample, app_m, RA_GW, dec_GW]
                rho_temp.append(rho_obs)
                z_temp.append(z_sample)
                Mz_temp.append(Mz_sample)
                RA_temp.append(RA_GW)
                dec_temp.append(dec_GW)
                m_app_temp.append(app_m)
                #print('N_out =  {} , Observed out =  {} '.format(N_out,len(app_m)))
            else:
                continue



            if N_out != len(inx_galaxies):    

                    N = N_in
                    index_galaxy = np.random.randint(0, len(z_cat), N)
                    z_sample = z_cat[index_galaxy]
                    app_m  = m_cat[index_galaxy]

                    #Skylocation
                    RA = RA_cat[index_galaxy]
                    dec = dec_cat[index_galaxy] 
                    RA_GW = RA
                    dec_GW = dec

                    dl = cosmology.z_to_dl_H_Omegas(z_sample,H0_sample, Omega_m_sample, 1 - Omega_m_sample)
                    chirp_sample = prior_mass.sample(len(z_sample))
                    Mz_sample = cosmology.M_z(chirp_sample,z_sample)

                    #Compute SNR 

                    rho_sample = cosmology.snr_Mz_z(Mz_sample, z_sample, H0_sample) 
                    rho_obs = np.sqrt((ncx2.rvs(2, rho_sample**2, size=len(index_galaxy))))
                    inx_obs = np.where(rho_obs > rth)[0]

                    if len(inx_obs): 

                        #observed
                        z_sample = (z_sample[inx_obs]) 
                        rho_obs  = (rho_obs[inx_obs])
                        Mz_sample = (Mz_sample[inx_obs])
                        app_m = (app_m[inx_obs])
                        RA_GW = (RA_GW[inx_obs])
                        dec_GW = (dec_GW[inx_obs])

                        rho_temp.append(rho_obs)
                        z_temp.append(z_sample)
                        Mz_temp.append(Mz_sample)
                        RA_temp.append(RA_GW)
                        dec_temp.append(dec_GW)
                        m_app_temp.append(app_m)


            
            
            if len(rho_temp) == 2 : 
                rho_temp = np.concatenate((rho_temp[0], rho_temp[1]), axis=0)
                z_temp = np.concatenate((z_temp[0], z_temp[1]), axis=0)
                Mz_temp = np.concatenate((Mz_temp[0], Mz_temp[1]), axis=0)
                RA_temp = np.concatenate((RA_temp[0], RA_temp[1]), axis=0)
                dec_temp = np.concatenate((dec_temp[0], dec_temp[1]), axis=0)
                m_app_temp = np.concatenate((m_app_temp[0], m_app_temp[1]), axis=0)

            elif len(rho_temp) == 0:
                continue

            else:
                rho_temp = rho_temp[0]
                z_temp = z_temp[0]
                Mz_temp = Mz_temp[0]
                RA_temp = RA_temp[0]
                dec_temp = dec_temp[0]
                m_app_temp = m_app_temp[0]

            if np.size(rho_temp) != 0:


                H0_list.append(round(float(H0_sample),3))
                Omega_m_list.append(round(float(Omega_m_sample),3))
                #print(z_temp)
                z_all = np.array(z_temp)
                pr_all = 1 / (1+z_all)
                draw_z = np.random.choice(z_all, 1, p = pr_all / np.sum(pr_all)) 
                z_sample = draw_z.item()
                inx_z = (np.where(z_all == z_sample)[0])[0]
                #list_parameters = np.array(list_parameters[inx_z, :])
                #print(rho_temp[0])
                rho_obs = rho_temp[inx_z]
                Mz_sample =Mz_temp[inx_z]
                app_m =m_app_temp[inx_z]
                RA_GW =RA_temp[inx_z]
                dec_GW =dec_temp[inx_z]

                rho_list.append(round(float(rho_obs),3))
                z_list.append(round(float(z_sample),3))
                Mz_list.append(round(float(Mz_sample),3))
                m_app_list.append(round(float(app_m),3))
                RA_list.append(round(float(RA_GW),3))
                dec_list.append(round(float(dec_GW),3))


                if i%batches == 0:
                    sys.stdout.write('\rEvents {}/{}, SNR = {}, Mz = {}, z = {}, m_app = {} , H0 = {}, Omega_m = {}'.format(i+1,
                                                                                                          Nevent,
                                                                                                          round(rho_obs,3),
                                                                                                          round(Mz_sample,3), 
                                                                                                          round(z_sample,3), 
                                                                                                          round(app_m,3),    
                                                                                                          round(H0_sample,3),   
                                                                                                          round(Omega_m_sample,3)))
                    print()
                i += 1
                break
            else: 
                continue

    end = time.time()
    passed_time = round(end - start,3)
    sys.stdout.write('\rEVENTS WERE GENERATED IN {} s ({} m , {} h)'.format(passed_time, round(passed_time/60,3), round(passed_time/3600,3)))
    return H0_list, Omega_m_list, rho_list, Mz_list, z_list, m_app_list, RA_list, dec_list


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
                                  'z': event[4], 'm_app': event[5],
                                  'RA': event[6], 'dec' : event[7]  })
        output_df.to_csv(path_data+'/data_mth_{}_SNRth_{}_{}_batch_{}.dat'.format(int(mth), int(rho_th), int(N/batches),j+Nbatch))
        j += 1
    if type_of_data == 'testing':
        #store events
        print('BATCH = {}'.format(j+1))
        event = GW_events(int(N/batches))
        output_df = pd.DataFrame({'H0': event[0], 'Omega_m' : event[1], 
                                  'SNR': event[2], 'Mz': event[3], 
                                  'z': event[4], 'm_app': event[5],
                                  'RA': event[6], 'dec' : event[7]  })
        output_df.to_csv(path_data+'/data_mth_{}_SNRth_{}_H0_70_Omega_0.3_1000.dat'.format(int(mth), int(rho_th), int(N/batches)))
        j += 1    
        
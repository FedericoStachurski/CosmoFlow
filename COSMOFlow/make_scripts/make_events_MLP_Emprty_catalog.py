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
print(parentdir)
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

from gw_functions import gw_priors
from gw_functions import gw_SNR


from scipy.stats import chi2
import numpy.random as rn
from scipy.stats import ncx2
from scipy.interpolate import splrep, splev
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import argparse
import random
np.random.seed(100)

from MultiLayerPerceptron.validate import run_on_dataset
from MultiLayerPerceptron.nn.model_creation import load_mlp
device = 'cpu'
model_name = 'SNR_approxiamator_2'
mlp = load_mlp(model_name, device, get_state_dict=True).to(device)
mlp.eval()


zmax = 2

type_of_data = 'testing'
SNRth = 8


#grid of z and M
M_grid =  np.linspace(-23,-5,100)
z_grid = np.linspace(0,zmax,100)


def draw_RA_Dec(N):
    #sample RA 
    ra_obs = np.random.uniform(0,360,N)
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

#spline p(z)
pz = evolution(z_grid)  * priors.p_z(z_grid, 0.3)
#pz = priors.p_z(z_grid, 0.3)


#intepolate z vs pz
spl_z = splrep(z_grid, pz )

def pz_int(z):
    return splev(z, spl_z )




if type_of_data == 'training':
    path_data = parentdir + r"/data_gwcosmo/empty_catalog/training_data_MLP/"
    N = 100000
    H0_samples = np.random.uniform(20,120,N)
    

if type_of_data == 'testing': 
    path_data = parentdir + r"/data_gwcosmo/empty_catalog/testing_data_MLP/"
    N = 250
    H0_samples = 70*np.ones(N)
    
    
#functions    
def SNR_from_MLP(GW_data):

    df = GW_data
    x_inds = [0,1,2]
    xdata = df.iloc[:,x_inds].to_numpy()
    xmeanstd = np.load(f'models/{model_name}/xdata_inputs.npy')
    net_out, time_tot, time_p_point = run_on_dataset(mlp,xdata,label_dim = None, 
                                                        device=device,y_transform_fn=None,runtime=True)
    pred = net_out
    pred[pred<0]=0
    snr_out = pred
    return snr_out






observed_snr = []
observed_H0 = []
observed_dl = [] 
observed_m1z = [] 
observed_m2z = []


# inx_in = np.where(H0_samples > 0)
H0 = H0_samples
while True:
    
    
    
    n = len(H0)
    #draw redshifts
    z = draw_cumulative_z(n, pz_int)
    
    

    #compute distance and apparent magnitudes
    dl = cosmology.fast_z_to_dl_v2(np.array(z),np.array(H0))

     

    #sample GW priors
    #m1, m2, _, _, _, _, _, _, _, _, _, _, _ = gw_priors.draw_prior(n)


    m1, m2 = np.random.uniform(5,50, size=(2, n))
    inx = np.where(m1 < m2)
    temp1 = m1[inx]
    temp2 = m2[inx]
    m1[inx] = temp2
    m2[inx] = temp1


    m1z = m1*(1+z)
    m2z = m2*(1+z)

    data_dict = {'dl': dl, 'm1z':m1z, 'm2z':m2z}
    GW_data = pd.DataFrame(data_dict)


    snrs  = SNR_from_MLP(GW_data)
    GW_data['snr_true'] = snrs

    
    
    snrs_obs = np.sqrt((ncx2.rvs(4, snrs**2, size=n, loc = 0, scale = 1)))
    GW_data['snr'] = snrs_obs
    inx_out = np.where((GW_data.snr_true != 0 ) & (GW_data.snr >= SNRth) & (GW_data.snr < 150) & (GW_data.dl < 35000))

    
  
    
    observed_snr.append(snrs_obs[inx_out])
    observed_H0.append(H0[inx_out])
    observed_dl.append(dl[inx_out])
    observed_m1z.append(m1z[inx_out])
    observed_m2z.append(m2z[inx_out])
    
    H0 = np.delete(H0, inx_out)
    
    
    Ndet = len(np.concatenate(observed_snr))
    
    sys.stdout.write('\r Detected events {}/{}. Percentage: {}%'.format(Ndet, N, int((Ndet/N)*100)))
#     sys.stdout.write(str(H0))
    if  Ndet == N: 
        break 
    
observed_snr = np.concatenate(observed_snr )
observed_H0 = np.concatenate(observed_H0 )
observed_dl = np.concatenate(observed_dl )
observed_m1z = np.concatenate(observed_m1z )
observed_m2z = np.concatenate(observed_m2z )




output_df = pd.DataFrame({'snr': observed_snr, 'H0': observed_H0, 'dl': observed_dl,
                          'm1': observed_m1z, 'm2': observed_m2z})
output_df.to_csv(path_data+'_data_{}_N_SNR_2.csv'.format(int(N)))






# #pass arguments 
# # Construct the argument parser
# ap = argparse.ArgumentParser()

# # Add the arguments to the parser
# ap.add_argument("-N", "--Nevents", required=True,
#    help="number of events", default=10000)
# ap.add_argument("-batches", "--batches", required=True,
#    help="number of batches", default=1)
# ap.add_argument("-type", "--type", required=True,
#    help="training or testing data?", default = 'training')
# ap.add_argument("-SNRth", "--threshold", required=True,
#    help="SNR threshold of GW event selection", default = 8)
# ap.add_argument("-zmax", "--maximum_redshift", required=True,
#    help="maximum redshift to sample from", default = 1)
# ap.add_argument("-H0", "--H0", required=False,
#    help="Hubble constant value", default = None)
# ap.add_argument("-Omega_m", "--Omega_m", required=False,
#    help="Matter density constant", default = None)
# ap.add_argument("-Nbatch", "--Nbatch", required=False,
#    help="batch number starting", default = 0)
# ap.add_argument("-fn", "--File_name", required=True,
#    help="File name", default = 0)



# args = vars(ap.parse_args())
# N = int(args['Nevents'])
# type_of_data = str(args['type'])
# rho_th = float(args['threshold'])
# zmax = float(args['maximum_redshift'])
# H0 = (args['H0'])
# Omega_m = (args['Omega_m'])
# batches = int(args['batches'])
# Nbatch = int(args['Nbatch'])
# filename = str(args['File_name'])

# print('EVENTS TO COLLECT = {}'.format(N))
# print('BATCHES = {}'.format(batches))
# print('TYPE = '+ type_of_data)
# print('SNRth = {}'.format(rho_th))
# print('Zmax = {}'.format(zmax))
# print('Omega_k = 0 ')

# if Omega_m is not None:
#     print('Omega_m = {}'.format(Omega_m))
# else:    
#     print('Omega_m ~ U[0,1]')
# if H0 is not None:
#     print('H0 = {}'.format(H0))
# else: 
#     print('H0 ~ U[20,120]')

# if Omega_m is not None:
#     print('Omega_m = {}'.format(Omega_m))
# else:    
#     print('Omega_m ~ U[0,1]')
# if H0 is not None:
#     print('H0 = {}'.format(H0))
# else: 
#     print('H0 ~ U[20,120]')


# zmax = zmax


# #grid of z and M
# M_grid =  np.linspace(-23,-5,100)
# z_grid = np.linspace(0,zmax,100)


# def draw_RA_Dec(N):
#     #sample RA 
#     ra_obs = np.random.uniform(0,360,N)
#     ra_obs = ra_obs* np.pi / 180
#     #sample declinations
#     P = np.random.uniform(0,1,N)
#     dec = np.arcsin(2*P-1) 
#     return ra_obs, dec


# def evolution(z):
#     lam = -1
#     return (1+z)**(lam)

# def draw_cumulative_z(N, distribution):
#     #grid = np.linspace(0.00001,zmax,100)
#     cdf = np.zeros(len(z_grid))
#     for i in range(len(z_grid )):
#         cdf[i] = quad(lambda z: distribution(z), z_grid[0], z_grid[i])[0]
#     cdf = cdf/np.max(cdf)  
#     t = rn.random(N)
#     samples = np.interp(t,cdf,z_grid)
#     return samples

# def draw_cumulative_M(N,H0, distribution):
#     #grid = np.linspace(-23,-5,100)
#     cdf = np.zeros(len(M_grid))
#     for i in range(len(M_grid)):
#         cdf[i] = quad(lambda M: distribution(M, H0),  M_grid [0], M_grid [i])[0]
#     cdf = cdf/np.max(cdf)     
#     t = rn.random(N)
#     samples = np.interp(t,cdf,M_grid)
#     return samples

# #spline p(z)
# pz = evolution(z_grid)  * priors.p_z(z_grid, 0.3)
# #pz = priors.p_z(z_grid, 0.3)


# #intepolate z vs pz
# spl_z = splrep(z_grid, pz )

# def pz_int(z):
#     return splev(z, spl_z )    



# if type_of_data == 'training':
#     path_data = r"/data_gwcosmo/empty_catalog/training_data/"
    

# if type_of_data == 'testing': 
#     path_data = r"/data_gwcosmo/empty_catalog/testing_data/"

# #%% GW event selection
# def GW_events(Nevent):
#     "GW event selection for analysis of Nevents"
    
#     start = time.time()
#     #store parameters in empty list name '__file__' is not defined
    
#     #GW_lists
#     dl_list, m1_list, m2_list, a1_list, a2_list, tilt1_list, tilt2_list, RA_list, dec_list, theta_jn_list =[],[],[],[],[],[],[],[],[],[]

#     #Cosmology_lists
#     z_list, H0_list, Omega_m_list= [], [], []
    
#     rth = rho_th

#     N = 100
#     i = 0
#     while i < Nevent:

#         #Sample H0 
#         if H0 is not None:
#             H0_sample  = float(H0) 
#         else: 
#             H0_sample = np.random.uniform(20,120, N)[0]

#         #Sample omega_m
#         if Omega_m is not None:
#             Omega_m_sample  = float(Omega_m)
#         else: 
#             Omega_m_sample = np.random.uniform(0,1, N)[0]
        

#         while True: 
            
#             #draw redshifts
#             z = draw_cumulative_z(N, pz_int)[0]
    
#             #compute distance and apparent magnitudes
#             dl = float(cosmology.fast_z_to_dl_v2(z,H0_sample))

#             #sample GW priors
#             m1, m2, a1, a2, tilt1, tilt2, RA, dec, theta_jn = gw_priors.draw_prior(N)
            
  
            
#             #compute SNR from injection using MLP
#             rho = gw_SNR.SNR_from_inj( dl, m1*(1+z), m2*(1+z), 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0)
            

#             if rho_obs > rho_th:
            
#                 #store all parameters 
# #                 RA_list.append(RA)
# #                 dec_list.append(dec)
#                 dl_list.append(dl)
#                 m1_list.append(m1*(1+z)) 
#                 m2_list.append(m2*(1+z)) 
# #                 a1_list.append(a1) 
# #                 a2_list.append(a2) 
# #                 tilt1_list.append(tilt1) 
# #                 tilt2_list.append(tilt2)
# #                 theta_jn_list.append(theta_jn)
                
                
# #                 z_list.append(z) 
#                 H0_list.append(H0_sample)


#                 i += 1 
#                 break
                
#         if i % 1 == 0:
#             sys.stdout.write('\r{}/{}, SNR = {}'.format(i, int(Nevent), rho_obs))
            
            
#         else: 
#             continue

#     end = time.time()
#     passed_time = round(end - start,3)
#     print()
#     print('EVENTS WERE GENERATED IN {} s ({} m , {} h)'.format(passed_time, round(passed_time/60,3),round(passed_time/3600,3)))
#     return H0_list,  dl_list, m1_list, m2_list #, a1_list, a2_list, tilt1_list, tilt2_list, RA_list, dec_list, theta_jn_list, z_list


# batches = batches

# os.chdir('..')
# new_path = os. getcwd()
# path_data = new_path + path_data


# for j in range(batches):
#     if type_of_data == 'training':
#         #store events
#         print('BATCH = {}'.format(j+1))
#         event = GW_events(int(N/batches))
#         output_df = pd.DataFrame({'H0': event[0], 'dl': event[1],
#                                   'm1': event[2], 'm2': event[3]})
#         output_df.to_csv(path_data+filename+'_data_{}_batch_{}.csv'.format(int(N/batches),j+Nbatch))
#         j += 1
#     if type_of_data == 'testing':
#         #store events
#         print('BATCH = {}'.format(j+1))
#         event = GW_events(int(N/batches))
#         output_df = pd.DataFrame({'H0': event[0], 'dl': event[1],
#                                   'm1': event[2], 'm2': event[3]})
#         output_df.to_csv(path_data+filename+
#                          '_data_{}.csv'.format(int(N/batches)))
#         j += 1   
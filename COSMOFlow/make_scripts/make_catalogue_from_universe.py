# Author: Federico Stachurski 
#Procedure: generate data frame of galaxies with corresponding redshift, RA, dec, luminosity distance, apparent magnitude
#           and absolute magnitude, given initial cosmological parameters and maximum distance in comoving volume. 
#Input: -N number of galaxies, -d max comoving distance, -H) hubble constant to use
#Output: data frame of galaxies
#Date: 16/07/2021

#%% import libraries
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np 
import scipy.stats
import matplotlib.pyplot as plt
import astropy.constants as const
from scipy.integrate import dblquad, quad
import numpy.random as rn
from tqdm import tqdm
from math import e
import random
from scipy.stats import multivariate_normal
np.random.seed(2)
import time
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.stats import loguniform
from cosmology_functions import priors 
from cosmology_functions import cosmology 
import argparse


# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-zmax", "--zmax", required=True,
   help="maximum redshift", default = 1)
ap.add_argument("-mth", "--mth", required=True,
   help="magnitude threshold", default = 24)
ap.add_argument("-H0", "--Hubble", required=True,
   help="Hubble constant to use")
ap.add_argument("-Omega_m", required=True,
   help="Baryonic matter density parameter")
 

args = vars(ap.parse_args())
zmax = float(args['zmax'])
mth = float(args['mth'])
H0 = float(args['Hubble']) #Hubble constant (km/s/Mpc)
omega_matter = float(args['Omega_m'])


path = "/data/wiay/federico/PhD/gwcosmoFLow_v2/catalogues/"
# set constants
c = const.c.to('km/s').value    #speed of light in km/s                         
h=H0/100                        #Hubble parameter 0.7
Omega_m = omega_matter
omega_lambda = 1- Omega_m
omega_k = 0

#Define number density 
n =(0.002)*(H0/50)**(3)
scale = (1/1e3)
scaled_n = n * scale


print('Generating universe with number density {} Mpc^-3, zmax = {}, H0 = {} km s^-1 Mpc^-1, omega_m = {}'.format(scaled_n, zmax, H0, Omega_m))

#Compute comoving volume 
Vcomoving = cosmology.Vc(zmax, H0, Omega_m)
Ngal = int(cosmology.Ngal(Vcomoving, scaled_n))

print('Number of glaxies in universe Ngal ={}'.format(Ngal))
      


# #%% initialise functions 
def z_to_dco(z,H, omega_m, omega_k, omega_lambda):
    "comoving distance as a function of z and H"
    def E(z):
        return np.sqrt((omega_m*(1+z)**(3) + omega_k*(1+z)**(2) + omega_lambda))
    
    def I(z):
        fact = lambda x: 1/E(x)
        integral = quad(fact, 0, z)
        return integral[0]
    if np.isscalar(z) is False:
        dco = np.zeros(len(z))    
        for i in range(len(z)): 
            dco[i]  =  (c / H) * I(z[i])
    else: 
        dco  =  (c / H) * I(z)
    return dco

      
      
max_comoving_dist =  z_to_dco(zmax, H0, Omega_m, omega_k, omega_lambda)       
      
print('d_comoving_max = {} Mpc'.format(round(max_comoving_dist, 2)))      
      
      
def dco_to_z(zmax, d, H, omega_m, omega_k, omega_lambda):
    "computes redshifts from a look up table using z_to_dco"
    z_grid = np.linspace(0,zmax,1000)
    dco = z_to_dco(z_grid,H, omega_m, omega_k, omega_lambda)
    redshifts = np.interp(d, dco, z_grid) 
    #v_pec=np.random.normal(0,200,len(d))
    obs_redshifts = redshifts  #(v_pec/c)*(1+redshifts)  
    return obs_redshifts

def luminosity_dist(d,z):
    "computes the luminsoity distance as distance x (1+z)"
    dl = d*(1+z) 
    return dl   

def app_mag(M,dl):
    m = M + 5*(np.log10(dl))+25
    return m.flatten()

# def p_M(M): #Schecter Function (B-band values)
#     phi = (1.61/100)*h**3
#     alpha = -1.21
#     Mc = -19.66 + 5*np.log10(h)
#     return (2/5)*phi*(np.log(10))*((10**((2/5)*(Mc-M)))**(alpha+1))*(e**(-10**((2/5)*(Mc-M))))

def draw_dist(N, dmax):
    print('Sampling Distances')
    def d_sample(n): 
        x_sample = np.random.uniform(-dmax,dmax, N)#np.random.multivariate_normal(x_cluster_means, x_cluster_cov,size=1)
        y_sample = np.random.uniform(-dmax,dmax, N)#np.random.multivariate_normal(y_cluster_means, y_cluster_cov,size=1)
        z_sample = np.random.uniform(-dmax,dmax, N)#np.random.multivariate_normal(z_cluster_means, z_cluster_cov,size=1)
        d_comoving = np.sqrt(x_sample**(2) + y_sample**(2) + z_sample**(2) )
        return d_comoving[d_comoving < dmax]

    d_com = d_sample(N)  
    temp =  len(d_com)
    while True: 
      #Shave the corners
        if temp < N: 
            rem = int(N - temp)
            d_com = np.hstack((d_com,d_sample(rem)))
            temp = temp + len(d_com)
        else:
            d_com = d_com[:N]
            break
    return d_com       


def draw_RA_Dec(N):
    #sample RA
    ra_obs = np.random.uniform(0,180,N)
    ra_obs = ra_obs* np.pi / 180
    #sample declinations
    P = np.random.uniform(0,1,N)
    dec = np.arcsin(2*P-1) 
    return ra_obs, dec


def draw_M(N):
    print('Sampling Absolute magnitudes from Schecter Function')
    M_grid = np.linspace(-23,-5,1000)
    cdf_M = np.zeros(len(M_grid))
    for i in range(len(M_grid )):
        cdf_M[i] = quad(lambda M: priors.p_M(M, H0), -23, M_grid[i])[0]
    cdf_M = cdf_M/np.max(cdf_M)     
    t = rn.random(N)
    M_sample = np.interp(t,cdf_M,M_grid)
    return M_sample

#%% run functions
start = time.time()
print('Sampling comoving distances')
comoving_dist = draw_dist(Ngal, max_comoving_dist)
print('Sampling RA and Dec')
RA_samples, dec_samples = draw_RA_Dec(Ngal)
print("Computing redshifts")
z_samples = dco_to_z(zmax, comoving_dist, H0, Omega_m, omega_k, omega_lambda)
print("Computing luminosity distances")
dl_samples = luminosity_dist(comoving_dist,z_samples)
M_samples = draw_M(Ngal)
print("Computing apparent magnitudes")
m_samples = app_mag(M_samples,dl_samples)
      
      
#get indicies of galaxies in catalogue
inx_cat = np.where(m_samples < mth)[0]

print('Number of galaxies in catalogue is Ncat = {} '.format(len(inx_cat)))      
      
RA_cat = (np.array(RA_samples))[inx_cat]
Dec_cat = (np.array(dec_samples))[inx_cat]
z_cat = (np.array(z_samples))[inx_cat]
m_cat = (np.array(m_samples))[inx_cat]
      
      
os.chdir('..')
os.chdir('..')
end = time.time()
print("Catalogue was generated in {} seconds ({} minutes)".format((end-start), (end-start)/60))
#print("In Catalogue there are Ncat/Ncat_TAO = " + str(len(z_cat)/884888)+' for a value of phi = '+str(scale))
#save universe as a data frame .dat file
print('Saving catalogue in Data Frame ')
output_df = pd.DataFrame({ 'z': z_cat, 'Apparent_m': m_cat, 'RA' :  RA_cat , 'dec' :  Dec_cat   } )  
output_df.to_csv(path+'catalogue_zmax_'+str(zmax)+'_mth_'+str(mth)+'.dat')  
print("Catalogue Completed")       

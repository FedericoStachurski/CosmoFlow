# Author: Federico Stachurski 
#Procedure: generate data frame of galaxies with corresponding redshift, RA, dec, luminosity distance, apparent magnitude
#           and absolute magnitude, given initial cosmological parameters and maximum distance in comoving volume. 
#Input: -N number of galaxies, -d max comoving distance, -H) hubble constant to use
#Output: data frame of galaxies
#Date: 11/12/2020

#%% import libraries 
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
import argparse

# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-N", "--Ngal", required=True,
   help="number of galaxies in universe")
ap.add_argument("-d", "--co_dist_max", required=True,
   help="maximum comoving distance to generate universe in Gpc", default = 1)
ap.add_argument("-H0", "--Hubble", required=True,
   help="Hubble constant to use")
ap.add_argument("-Omega_m", required=True,
   help="Baryonic matter density parameter")
ap.add_argument("-Name", "--Universe_Name", required=True,
   help="Filename to save dataframe")   

args = vars(ap.parse_args())
N = int(args['Ngal'])
dmax = float(args['co_dist_max'])
H0 = float(args['Hubble']) #Hubble constant (km/s/Mpc)
omega_matter = float(args['Omega_m'])
Name = args['Universe_Name']

path = "universes/"
# set constants
c = const.c.to('km/s').value    #speed of light in km/s                         
h=H0/100                        #Hubble parameter 0.7
dmax = dmax*1000                #Mpc
omega_m = omega_matter
omega_lambda = 1- omega_m
omega_k = 0

print('Generating universe with {} galaxies, max comoving distance of {} Mpc and H0 = {}, Omega_m = {}'.format(N, dmax, H0, omega_m))

#%% initialise functions 
def z_to_dl(z,H, omega_m, omega_k, omega_lambda):
    "distance as a function of z and H"
    def E(z):
        return np.sqrt((omega_m*(1+z)**(3) + omega_k*(1+z)**(2) + omega_lambda))
    
    def I(z):
        fact = lambda x: 1/E(x)
        integral = quad(fact, 0, z)
        return integral[0]
    dl = np.zeros(len(z))    
    for i in range(len(z)): 
        dl[i]  =  (c / H) * I(z[i])
    return dl

def dl_to_z(d, H, omega_m, omega_k, omega_lambda):
    "computes redshifts from a look up table using z_to_dl"
    z_grid = np.linspace(0,2,1000)
    dl = z_to_dl(z_grid,H, omega_m, omega_k, omega_lambda)
    redshifts = np.interp(d, dl, z_grid) 
    #v_pec=np.random.normal(0,200,len(d))
    obs_redshifts = redshifts  #(v_pec/c)*(1+redshifts)  
    return obs_redshifts

def luminosity_dist(d,z):
    "computes the luminsoity distance as distance x (1+z)"
    dl = d*(1+z) 
    return dl   

def app_mag(M,dl):
    print('Computing apparent magnitudes from absolute Magnitudes and luminosity distances')
    m = M + 5*(np.log10(dl))+25
    return m.flatten()

def p_M(M): #Schecter Function (B-band values)
    phi = (1.61/100)*h**3
    alpha = -1.21
    Mc = -19.66 + 5*np.log10(h)
    return (2/5)*phi*(np.log(10))*((10**((2/5)*(Mc-M)))**(alpha+1))*(e**(-10**((2/5)*(Mc-M))))

def draw_dist(N):
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
        if temp < N: 
            rem = int(N - temp)
            d_com = np.hstack((d_com,d_sample(rem)))
            temp = temp + len(d_com)
        else:
            d_com = d_com[:N]
            break
    return d_com       


def draw_RA_Dec(N):
    print('Sampling Ra and Dec')
    #sample RA
    ra_obs = np.random.uniform(0,180,N)

    #sample declinations
    P = np.random.uniform(0,1,N)
    dec = np.arcsin(2*P-1) 
    return ra_obs, dec


def draw_M(N):
    print('Sampling Absolute magnitudes from Schecter Function')
    M_grid = np.linspace(-23,-5,10000)
    cdf_M = np.zeros(len(M_grid))
    for i in range(len(M_grid )):
        cdf_M[i] = quad(lambda z: p_M(z), -23, M_grid[i])[0]
    cdf_M = cdf_M/np.max(cdf_M)     
    t = rn.random(N)
    M_sample = np.interp(t,cdf_M,M_grid)
    return M_sample

#%% run functions
start = time.time()
comoving_dist = draw_dist(N)
RA_samples, dec_samples = draw_RA_Dec(N)
print("Computing redshifts")
z_samples = dl_to_z(comoving_dist,H0, omega_m, omega_k, omega_lambda)
print("Computing luminosity distances")
dl_samples = luminosity_dist(comoving_dist,z_samples)
M_samples = draw_M(N)
print("Computing apparent magnitudes")
m_samples = app_mag(M_samples,dl_samples)

end = time.time()
print("Universe was generated in {} seconds ({} minutes)".format((end-start), (end-start)/60))
#save universe as a data frame .dat file
print('Saving galaxies in Data Frame ')
output_df = pd.DataFrame({ 'z': z_samples, 'Apparent_m': m_samples, 'RA' :  RA_samples , 'dec' :  dec_samples   } )  
output_df.to_csv(path+ Name +'.dat')  
print("Universe Complete")       

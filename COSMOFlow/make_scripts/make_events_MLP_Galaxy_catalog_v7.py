#Author: Federico Stachurski 
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

import time
from scipy.stats import chi2
import numpy.random as rn
from scipy.stats import ncx2
from scipy.interpolate import splrep, splev
from scipy.stats import truncnorm
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import argparse
import random
import h5py 
import healpy as hp
import multiprocessing


from MultiLayerPerceptron.validate import run_on_dataset
from MultiLayerPerceptron.nn.model_creation import load_mlp
device = 'cpu'
model_name = 'SNR_approxiamator_full_para_HA_v5'
mlp = load_mlp(model_name, device, get_state_dict=True).to(device)
mlp.eval()

zmax = 1.2
zmin = 0.0001
NSIDE = 32
type_of_data = 'training'
in_out = True
SNRth = 11
NSIDE = 32
Nselect = 5
N = 250_000
distributions = {'mass':'PowerLaw+Peak'}
threads = 10
band = 'Bj'
# Mabs_min = -27.00
# Mabs_max = -19.00


np.random.seed(101022)

#Load  pixelated galaxy catalog
Npix = hp.nside2npix(NSIDE)

if in_out is True:
    def load_cat_by_pix(pix):
        loaded_pix = pd.read_csv('/data/wiay/federico/PhD/cosmoflow/COSMOFlow/pixelated_catalogs/GLADE+_pix/pixel_{}'.format(pix))
    #     loaded_pix = loaded_pix.dropna()
        return loaded_pix

    with multiprocessing.Pool(threads) as p:
        catalog_pixelated = list(tqdm(p.imap(load_cat_by_pix,np.arange(Npix)), total = Npix, desc = 'Loading mth map, NSIDE = {}'.format(NSIDE)))



#grid of z and M

z_grid = np.linspace(zmin,zmax,100)

def HA(time, RA):
    def time2angle(time):
        time = time / 3600.0
        return time*(15/1)*(np.pi/180)
    LHA = (time2angle(time) - RA)*(180/np.pi)

    return LHA

def round_base(x, base=100):
    return int(base * round(float(x)/base))

def draw_RA_Dec(N):
    #sample RA 
    ra_obs = np.random.uniform(0,360,N)
    ra_obs = ra_obs* np.pi / 180
    #sample declinations
    P = np.random.uniform(0,1,N)
    dec = np.arcsin(2*P-1) 
    return ra_obs, dec


def Madau_factor(z, gamma = 4.59, k = 2.86, zp = 2.47):
    num = (1+z)**(gamma - 1)
    den = 1 + ((1+z)/(1+zp))**(gamma+k)
    return num/den


Mabs_min = -22.00
Mabs_max = -16.50


def M_grid_H0(low,high, H0):
    return np.linspace(high  + 5*np.log10(H0/100),low  + 5*np.log10(H0/100), 100)


def LF_weight_L(M,H0, phi_star =1.16*10**(-2), alpha = -1.21 + 1, Mstar = -19.66):
    Mstar = Mstar + 5*np.log10(H0/100)
    phi_star = phi_star * (H0/100)**(3)
    return phi_star * 0.4*np.log(10)*10**(-0.4*(M - Mstar)*(alpha + 1))*np.exp(-10**(-0.4*(M - Mstar)))


#spline p(z)
pz = Madau_factor(z_grid)  * priors.p_z(z_grid, omega_m = 0.3065)
#pz = priors.p_z(z_grid, 0.3)


#intepolate z vs pz
spl_z = splrep(z_grid, pz )

def pz_int(z):
    return splev(z, spl_z )

#grid = np.linspace(0.00001,zmax,100)
cdf_z = np.zeros(len(z_grid))
for i in range(len(z_grid )):
    cdf_z[i] = quad(lambda z: pz_int(z), z_grid[0], z_grid[i])[0]
cdf_z = cdf_z/np.max(cdf_z)  


def draw_cumulative_z(N):
    t = rn.random(N)
    samples = np.interp(t,cdf_z,z_grid)
    return samples



def sample_M_H0(H0):
    M_grid = M_grid_H0(-22, -16.5, H0)
    def cdf_M(H0):
        cdf = np.zeros(len(M_grid))
        for i in range(len(M_grid)):
            cdf[i] = quad(lambda M: LF_weight_L(M, H0),  M_grid [-1], M_grid [-(i+1)])[0] ## Luminosity weighting 
        return cdf/np.max(cdf)    

    def sample_M_from_cdf(cdf, N = 1):
        t = rn.random(N)
        samples = np.interp(t,cdf,M_grid)
        return samples
    cdf = cdf_M(H0)
    return sample_M_from_cdf(cdf)





def mag2lum(M):
    return 10**(M/(-2.5))


map_mth = np.loadtxt('/data/wiay/federico/PhD/cosmoflow/COSMOFlow/magnitude_threshold_maps/NSIDE_32_mth_map_GLADE_{}.txt'.format(band))

def mth_from_RAdec(NSIDE, RA, dec, map_mth):
    phi = np.array(np.deg2rad(np.rad2deg(RA)))
    theta = np.pi/2 - np.array(np.deg2rad(np.rad2deg(dec)))
    pix_inx = hp.ang2pix(NSIDE, theta, phi)
    return map_mth[pix_inx]

def pix_from_RAdec(NSIDE, RA, dec):
    phi = np.array(np.deg2rad(np.rad2deg(RA)))
    theta = np.pi/2 - np.array(np.deg2rad(np.rad2deg(dec)))
    pix_inx = hp.ang2pix(NSIDE, theta, phi)
    return pix_inx
#Compute pixel index for galaxies 

#pixel_indicies_galaxies = pix_from_RAdec(NSIDE, catalog.ra, catalog.dec)


def load_pixel(pix):
    loaded_pix = catalog_pixelated[pix]
    return loaded_pix, Npix


def select_gal_from_pix(pixels_H0s):
    pixel, H0 = pixels_H0s
    loaded_pixel, Ngalpix = load_pixel(int(pixel))
    loaded_pixel = loaded_pixel[['z','RA','dec', 'sigmaz', 'm'+band]]
    loaded_pixel = loaded_pixel.dropna()
    loaded_pixel = loaded_pixel[loaded_pixel.z <= zmax]
    loaded_pixel = loaded_pixel[loaded_pixel.z >= zmin]
    loaded_pixel['RA'] = np.deg2rad(loaded_pixel['RA'])
    loaded_pixel['dec'] = np.deg2rad(loaded_pixel['dec'])

    Ngalpix = len(loaded_pixel)
    
    if loaded_pixel.empty is False:
        z_gal_selected = loaded_pixel.z 
        repeated_H0_in_pix = np.ones(Ngalpix)*H0
        dl_galaxies = cosmology.fast_z_to_dl_v2(np.array(z_gal_selected).flatten(),np.array(repeated_H0_in_pix).flatten())
        #get luminsoity
        absolute_mag = cosmology.abs_M(loaded_pixel['m'+band],dl_galaxies)
  
        luminosities = mag2lum(absolute_mag)
        weights_gal = luminosities * Madau_factor(z_gal_selected)
        weights_gal /= np.sum(weights_gal)
        gal_id = np.random.choice(np.arange(Ngalpix), size = 1, p = weights_gal)
        return loaded_pixel.iloc[gal_id,:]



if type_of_data == 'training':
    if in_out is True: 
        path_data = parentdir + r"/data_gwcosmo/galaxy_catalog/training_data_from_MLP/"
    else:
        path_data = parentdir + r"/data_gwcosmo/empty_catalog/training_data_from_MLP/"
    H0_samples = np.random.uniform(20,140,N)
    M = [] 
    if in_out is True:
        for i in tqdm(range(N), desc='Computing CDFs for Schechter Function'):
            M.append(sample_M_H0(H0_samples[i]))
        M = np.array(M)
    else: 
        M = np.M = np.random.uniform(0,1,N)

if type_of_data == 'testing': 
    if in_out is True: 
        path_data = parentdir + r"/data_gwcosmo/galaxy_catalog/training_data_from_MLP/"
    else:
        path_data = parentdir + r"/data_gwcosmo/empty_catalog/training_data_from_MLP/"
    N = 250
    H0_samples = 70*np.ones(N)
    R_nums = np.random.uniform(0,1, size = N)
    M = [] 
    if in_out is True:
        for i in tqdm(range(N), desc='Computing CDFs for Schechter Function'):
            M.append(sample_M_H0(H0_samples[i]))
        M = np.array(M)
    else: 
        M = np.random.uniform(0,1,N)
    
    
#functions    
def SNR_from_MLP(GW_data):

    df = GW_data
    x_inds = [0, 1,2, 3, 4, 5, 6, 7, 8, 9,10, 11, 12]
    xdata = df.iloc[:,x_inds].to_numpy()
    xmeanstd = np.load(f'models/{model_name}/xdata_inputs.npy')
    net_out, time_tot, time_p_point = run_on_dataset(mlp,xdata,label_dim = None, 
                                                        device=device,y_transform_fn=None,runtime=True)
    pred = net_out
    
    pred = np.exp(pred) #/np.array(df.dl)
    snr_out = pred
    return snr_out

H0 = H0_samples
N_missed = N
list_data = []
missed_H0 = H0
missed_M = M
if type_of_data =='testing':
    missed_R = R_nums

while True:    
    st = time.perf_counter()
    start = time.time()
    n = len(missed_H0)

    inx_gal = np.zeros(n)
    #draw redshifts
    z = draw_cumulative_z(n)
#     repeated_H0 = np.repeat(missed_H0, int(select))
#     repeated_M = np.repeat(missed_M, int(select))
    #compute distance and apparent magnitudes
    dl = cosmology.fast_z_to_dl_v2(np.array(z),np.array(missed_H0))
    #Make sure all are arrays
    z = np.array(z)
    dl = np.array(dl)
    RA, dec = draw_RA_Dec(int(n))
    if in_out is True: 
        app_samples = cosmology.app_mag(missed_M.flatten(),dl.flatten())
    else: 
        app_samples = np.ones(n)
    end = time.time()
    print('Dl, Ra, dec  sampled, took = {} s'.format(round(end - start, 3)))
    

    if in_out is True:
        #Magnitude threshold map
        mth_list = np.array([mth_from_RAdec(NSIDE, RA, dec, map_mth)]).flatten()
        pix_list = np.array([pix_from_RAdec(NSIDE, RA, dec)]).flatten()
        inx_in_gal = np.where((app_samples < mth_list) == True)[0] 
        print('Loading {} pixels'.format(len(inx_in_gal)))
    else:   
        inx_in_gal = np.where((app_samples < 0) == True)[0] #{NxNselect}

    if len(inx_in_gal) > 0:

        #Add luminosity weights in the future 
        #Random choice glaaxy id with weights 
        pix_list = np.array(pix_list[inx_in_gal])
        H0_in_list = np.array(missed_H0[inx_in_gal])
        pixel_H0 = np.array([pix_list, H0_in_list]).T

        
        start = time.time()
        with multiprocessing.Pool(threads) as p:
            selected_cat_pixels = list(p.imap(select_gal_from_pix,pixel_H0))   
    
        gal_selected = pd.concat(selected_cat_pixels)
        RA_gal = np.array(gal_selected.RA)
        dec_gal = np.array(gal_selected.dec)
        z_true_gal = np.array(gal_selected.z)
        sigmaz_gal = np.array(gal_selected.sigmaz)
        #z_obs_gal = np.random.normal(z_true_gal, sigmaz_gal)
        a, b = (zmin - z_true_gal) / sigmaz_gal, (zmax - z_true_gal) / sigmaz_gal
        z_obs_gal = truncnorm.rvs(a, b, loc=z_true_gal, scale=abs(sigmaz_gal), size=len(z_true_gal))
        m_obs_gal = np.array(gal_selected['m'+band])

        dl_gal = cosmology.fast_z_to_dl_v2(np.array(z_obs_gal),np.array(H0_in_list))

        #Switch z values in z array with zgal and dgal
        z[inx_in_gal] = z_obs_gal
        dl[inx_in_gal] = dl_gal 
        RA[inx_in_gal] = RA_gal
        dec[inx_in_gal] = dec_gal
        app_samples[inx_in_gal] = m_obs_gal
        inx_gal[inx_in_gal] = 1
    
        end = time.time()
    print('GLADE+ catalog selected, took = {} s'.format(round(end - start, 3)))
    
    
    select = int(Nselect*(N/N_missed)**1)
    if type_of_data == 'testing':
        repeated_Rnums = np.repeat(missed_R, select) 
    #repeat location parameters nselect times
    repeated_H0 = np.repeat(missed_H0, select) 
    repeated_M = np.repeat(missed_M, select)
    repeated_z = np.repeat(z, select)   
    repeated_dl = np.repeat(dl, select)   
    repeated_RA = np.repeat(RA, select) 
    repeated_dec = np.repeat(dec, select)
    repeated_app_mag = np.repeat(app_samples, select) 
    repeated_inx_gal = np.repeat(inx_gal, select) 
    
    nxN = int(n*select)
    
    start = time.time()
    _, m1, m2, a1, a2, tilt1, tilt2, _, _, theta_jn, phi_jl, phi_12,psi, _ , geo_time = gw_priors.draw_prior(int(nxN), distributions)
    end = time.time()
    print('GW para sampled, took = {} s'.format(round(end - start, 3)))
    m1z = m1*(1+repeated_z)
    m2z = m2*(1+repeated_z)
    
    #Compute HA using RA and geo_time
    ha = np.array(HA(geo_time, repeated_RA - np.pi))
    
    MLP_data_dict = {'dl':repeated_dl, 'm1':m1z, 'm2':m2z,'a1': a1, 'a2': a2,
                 'tilt1': tilt1, 'tilt2': tilt2,'HA':ha, 'dec':repeated_dec,
                 'theta_jn':theta_jn, 'phi_jl':phi_jl, 'phi_12':phi_12, 'polarization':psi }   
    
    data_dict = {'dl':repeated_dl, 'm1':m1z, 'm2':m2z,'a1': a1, 'a2': a2,
             'tilt1': tilt1, 'tilt2': tilt2,'RA':repeated_RA, 'dec':repeated_dec,
             'theta_jn':theta_jn, 'phi_jl':phi_jl, 'phi_12':phi_12, 'polarization':psi , 'geo_time':geo_time}   
    
    MLP_GW_data = pd.DataFrame(MLP_data_dict)
    GW_data = pd.DataFrame(data_dict)
    
           
    snrs  = SNR_from_MLP(MLP_GW_data)
    snrs_obs = np.sqrt((ncx2.rvs(6, snrs**2, size=nxN, loc = 0, scale = 1)))        
    et = time.perf_counter()
    print('Nselect = {}'.format(int(select)))
    print('Time:{}'.format(et-st))
    GW_data['snr'] = snrs_obs 
    inx_out = np.where((GW_data.snr >= SNRth))[0]   
    #print(inx_out)
    GW_data['H0'] = repeated_H0
    GW_data['M'] = repeated_M
    GW_data['app_mag'] = repeated_app_mag
    GW_data['inx'] = repeated_inx_gal
    
    if type_of_data =='testing':
        GW_data['R'] = repeated_Rnums
        
    inds_to_keep = []
    for k in range(n):
        try:
            inds_to_keep.append(inx_out[(k*int(select) < inx_out) & (inx_out < (k+1)*int(select))][0])
        except IndexError:
            pass
    if len(inds_to_keep) == 0:
        continue
        
    out_data = GW_data.loc[np.array(inds_to_keep)]
    #print(out_data.snr)
#     print(repeated_H0)
    list_data.append(out_data)
    #print(len(pd.concat(list_data)))
#     print(len(repeated_H0))
    
    if type_of_data =='training':
        missed_H0 = np.setxor1d(out_data['H0'].to_numpy(),repeated_H0)   
        missed_M = np.setxor1d(out_data['M'].to_numpy(),repeated_M)
        #print(missed_H0)
        N_missed = len(missed_H0)
        print('H0 that we missed:{}'.format(N_missed))         

        if len(missed_H0) == 0 : 
            repeated_H0 = np.repeat(missed_H0, select)
            break
            
    elif type_of_data =='testing':
        missed_R = np.setxor1d(out_data['R'].to_numpy(),repeated_Rnums)
        
        temp = []
        for x in R_nums:
            temp.append(np.where(missed_R == x)[0])
            
        indices_R = np.concatenate(temp,axis=0)
        #indices_R = [np.where(missed_R==x)[0][0] for x in R_nums]
        missed_H0 = missed_H0[indices_R]
        missed_M = missed_M[indices_R]

        N_missed = len(missed_H0)
        print('H0 that we missed:', N_missed)
        if N_missed == 0 : 
            repeated_H0 = np.repeat(missed_H0, select)
            break
        
        
        
        
        
        
GW_data = pd.concat(list_data)    
output_df = GW_data[['snr', 'H0', 'dl', 'm1', 'm2', 'RA', 'dec',
                     'a1', 'a2', 'tilt1', 'tilt2', 'theta_jn',
                     'phi_jl', 'phi_12', 'polarization','geo_time', 'app_mag', 'inx']]
output_df.to_csv(path_data+'Bjband_batch_4_{}_N_SNR_{}_Nelect_{}__Full_para_v2.csv'.format(int(N), int(SNRth), int(Nselect)))
    
  
    


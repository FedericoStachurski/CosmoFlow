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
np.random.seed(1)

from MultiLayerPerceptron.validate import run_on_dataset
from MultiLayerPerceptron.nn.model_creation import load_mlp
device = 'cpu'
model_name = 'SNR_approxiamator_full_para_HA_v5'
mlp = load_mlp(model_name, device, get_state_dict=True).to(device)
mlp.eval()

#load Glade+
pathname = '/data/wiay/galaxy_catalogs'
catalog_file = 'glade+.hdf5'

with h5py.File(pathname+'/'+catalog_file, "r") as f:
    # List all columns
    columns = list(f.keys())
    
    z = f.get('z')[:]
    sigmaz = f.get('sigmaz')[:] 
    ra = f.get('ra')[:] 
    dec = f.get('dec')[:] 
    m_B = f.get('m_B')[:]

print(columns)
dic = {'z': z , 'sigmaz': sigmaz , 'ra':ra, 'dec':dec, 'm_B':m_B}
catalog = pd.DataFrame(dic)   
weights = 1 / (1+catalog.z)
weights /= np.sum(weights)
galaxy_ids = np.linspace(0,len(catalog)-1, len(catalog))


zmax = 2.0
zmin = 0.0001
NSIDE = 32
type_of_data = 'training'
SNRth = 10



#grid of z and M
M_grid =  np.linspace(-23,-5,25)
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


def evolution(z):
    lam = -1
    return (1+z)**(lam)


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

#grid = np.linspace(0.00001,zmax,100)
cdf_z = np.zeros(len(z_grid))
for i in range(len(z_grid )):
    cdf_z[i] = quad(lambda z: pz_int(z), z_grid[0], z_grid[i])[0]
cdf_z = cdf_z/np.max(cdf_z)  


def draw_cumulative_z(N):
    t = rn.random(N)
    samples = np.interp(t,cdf_z,z_grid)
    return samples

def cdf_M(H0):
    para_dict ={'phi': 1.6*(10**-2)*(H0/100)**(3), 'alpha': -1.07, 'Mc': -20.47 + 5*np.log10(H0/100)}
    cdf = np.zeros(len(M_grid))
    for i in range(len(M_grid)):
        cdf[i] = quad(lambda M: priors.p_M_weight_L(M, H0, para_dict),  M_grid [0], M_grid [i])[0]
    return cdf/np.max(cdf)    

def sample_M_from_cdf(cdf, N):
    t = rn.random(N)
    samples = np.interp(t,cdf,M_grid)
    return samples



def mag2lum(M):
    return 10**(M/(-2.512))


map_mth = np.loadtxt('/data/wiay/federico/PhD/cosmoflow/COSMOFlow/magnitude_threshold_maps/NSIDE_32_mth_map_GLADE.txt')

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

pixel_indicies_galaxies = pix_from_RAdec(NSIDE, catalog.ra, catalog.dec)


if type_of_data == 'training':
    path_data = parentdir + r"/data_gwcosmo/galaxy_catalog/training_data_from_MLP/"
    N = 50_000
    H0_samples = np.random.uniform(20,120,N)
    cdfs = np.ones((N,25))
    for i in tqdm(range(N), desc='Computing CDFs for Schechter Function'):
        cdfs[i, :] = cdf_M(H0_samples[i])
    
    

if type_of_data == 'testing': 
    path_data = parentdir + r"/data_gwcosmo/galaxy_catalog/testing_data_from_MLP/"
    N = 250
    H0_samples = 70*np.ones(N)
    cdfs = np.ones((N,25))
    for i in tqdm(range(N), desc='Computing CDFs for Schechter Function'):
        cdfs[i, :] = cdf_M(H0_samples[i])
    
    
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


# inx_in = np.where(H0_samples > 0)
H0 = H0_samples


Nselect = 1*10**(2) 
N_missed = N

M = [] 
for cdf in cdfs:
    M_val = sample_M_from_cdf(cdf, 1)
#     lum_weigths = mag2lum(M_arr_temp) / np.sum(mag2lum(M_arr_temp))
#     M_val = np.random.choice(M_arr_temp, size = 1, p=lum_weigths)    
    M.append(M_val)
    
#M = np.array(M)  #nxNselect


list_data = []
missed_H0 = H0
missed_M = M

while True:    
    
    n = len(missed_H0)
    select = round_base(Nselect*N/N_missed , base = Nselect)
    nxN = int(n*select)
    inx_gal = np.zeros(nxN)
    #draw redshifts
    z = draw_cumulative_z(nxN)
    repeated_H0 = np.repeat(missed_H0, int(select))
    repeated_M = np.repeat(missed_M, int(select))
    #compute distance and apparent magnitudes
    dl = cosmology.fast_z_to_dl_v2(np.array(z),np.array(repeated_H0 ))
    #Make sure all are arrays
    z = np.array(z)
    dl = np.array(dl)
    print('Dl sampled')    
    distributions = {'mass':'Power-law'}
    _, m1, m2, a1, a2, tilt1, tilt2, RA, dec, theta_jn, phi_jl, phi_12,psi, _ , geo_time = gw_priors.draw_prior(nxN, distributions)
    print('GW para sampled')
    app_samples = cosmology.app_mag(repeated_M.flatten(),dl.flatten())
    
    
    #Magnitude threshold map
    mth_list = np.array([mth_from_RAdec(NSIDE, RA, dec, map_mth)])
    pix_list = np.array([pix_from_RAdec(NSIDE, RA, dec)]) 
    inx_in_gal = np.where((app_samples < mth_list) == True)[1]
    #inx_out_gal = np.where((app_mag < mth_list) == False )
#     print(len(inx_in_gal))
#     print(len(pix_list))
    pix_list = pix_list[0][inx_in_gal]
    H0_in_list = repeated_H0[inx_in_gal]
    pix_selected_inx = np.where(np.in1d(pixel_indicies_galaxies, pix_list))[0]
    sorted_indx = np.argsort(pix_list)
    sorted_pix = pix_list[sorted_indx]
    H0_in_list = H0_in_list[sorted_indx]
    
    
    #print(set(sorted_pix).issubset(set(pixel_indicies_galaxies)))
    
    #Check how many times to repeat H0 
    idx = np.searchsorted(sorted_pix,pixel_indicies_galaxies)
    idx[idx==len(sorted_pix)] = 0
    mask = np.array(sorted_pix)[idx]==np.array(pixel_indicies_galaxies)
    out = np.bincount(idx[mask])
    
#     print(len(sorted_pix))
#     print(len(H0_in_list))
#     print(len(out))
    
    H0_in_list = np.repeat(H0_in_list, out)
    
    
#     print(len(pix_selected_inx))
#     print(len(inx_in_gal))

    if len(inx_in_gal) > 0:
        #print(len(inx_in_gal))
        #Add luminosity weights in the future 
        #Random choice glaaxy id with weights 
        
        galaxy_ids_selected = galaxy_ids[pix_selected_inx]
        z_gal_selected = catalog.z[pix_selected_inx]
        dl_galaxies = cosmology.fast_z_to_dl_v2(z_gal_selected,H0_in_list)
        
        #get luminsoity
        absolute_mag = cosmology.abs_M(catalog.m_B[pix_selected_inx],dl_galaxies)
        luminosities = mag2lum(absolute_mag)
        
        weights_gal = luminosities / (1+z_gal_selected)
        #print(weights_gal)
        weights_gal[np.isnan(weights_gal)] = 0
        weights_gal /= np.sum(weights_gal)
        
        
        gal_id = np.random.choice(galaxy_ids_selected, size = len(inx_in_gal), p = weights_gal)
        gal_selected = catalog.iloc[gal_id,:]

        RA_gal = np.array(gal_selected.ra)
        dec_gal = np.array(gal_selected.dec)
        z_true_gal = np.array(gal_selected.z)
        sigmaz_gal = np.array(gal_selected.sigmaz)
        #z_obs_gal = np.random.normal(z_true_gal, sigmaz_gal)
        a, b = (zmin - z_true_gal) / sigmaz_gal, (zmax - z_true_gal) / sigmaz_gal
        z_obs_gal = truncnorm.rvs(a, b, loc=z_true_gal, scale=abs(sigmaz_gal), size=len(sigmaz_gal))
        m_obs_gal = np.array(gal_selected.m_B)

        dl_gal = cosmology.fast_z_to_dl_v2(np.array(z_obs_gal),np.array(repeated_H0[inx_in_gal]))

        #Switch z values in z array with zgal and dgal
        z[inx_in_gal] = z_obs_gal
        dl[inx_in_gal] = dl_gal 
        RA[inx_in_gal] = RA_gal
        dec[inx_in_gal] = dec_gal
        app_samples[inx_in_gal] = m_obs_gal
        inx_gal[inx_in_gal] = 1
    
    #print(np.where(inx_gal == 1))
    
    m1z = m1*(1+z)
    m2z = m2*(1+z)
    
    #Compute HA using RA and geo_time
    ha = np.array(HA(geo_time, RA - np.pi))
    
    MLP_data_dict = {'dl':dl, 'm1':m1z, 'm2':m2z,'a1': a1, 'a2': a2,
                 'tilt1': tilt1, 'tilt2': tilt2,'HA':ha, 'dec':dec,
                 'theta_jn':theta_jn, 'phi_jl':phi_jl, 'phi_12':phi_12, 'polarization':psi }   
    
    data_dict = {'dl':dl, 'm1':m1z, 'm2':m2z,'a1': a1, 'a2': a2,
             'tilt1': tilt1, 'tilt2': tilt2,'RA':RA, 'dec':dec,
             'theta_jn':theta_jn, 'phi_jl':phi_jl, 'phi_12':phi_12, 'polarization':psi , 'geo_time':geo_time}   
    
    MLP_GW_data = pd.DataFrame(MLP_data_dict)
    GW_data = pd.DataFrame(data_dict)
    
    st = time.perf_counter()        
    snrs  = SNR_from_MLP(MLP_GW_data)
    snrs_obs = np.sqrt((ncx2.rvs(6, snrs**2, size=nxN, loc = 0, scale = 1)))        
    et = time.perf_counter()
    print('Time:',et-st)
    GW_data['snr'] = snrs_obs 
    inx_out = np.where((GW_data.snr >= SNRth) & (GW_data.snr < 150) & (GW_data.dl < 15_000))[0]            
    GW_data['H0'] = repeated_H0
    GW_data['M'] = repeated_M
    GW_data['app_mag'] = app_samples
    GW_data['inx'] = inx_gal
    inds_to_keep = []
    for k in range(n):
        try:
            inds_to_keep.append(inx_out[(k*int(select) < inx_out) & (inx_out < (k+1)*int(select))][0])
        except IndexError:
            pass
    if len(inds_to_keep) == 0:
        continue
        
    out_data = GW_data.loc[np.array(inds_to_keep)]
    list_data.append(out_data)
    N_missed = len(np.setxor1d(out_data['H0'].to_numpy(),repeated_H0))
    print('H0 that we missed:', N_missed)    
    missed_H0 = np.setxor1d(out_data['H0'].to_numpy(),repeated_H0)   
    missed_M = np.setxor1d(out_data['M'].to_numpy(),repeated_M) 
    if len(missed_H0) == 0 : 
        break
        
GW_data = pd.concat(list_data)    
output_df = GW_data[['snr', 'H0', 'dl', 'm1', 'm2', 'RA', 'dec',
                     'a1', 'a2', 'tilt1', 'tilt2', 'theta_jn',
                     'phi_jl', 'phi_12', 'polarization','geo_time', 'app_mag', 'inx']]
output_df.to_csv(path_data+'_data_{}_N_SNR_{}_Nelect_{}__Full_para_v2.csv'.format(int(N), int(SNRth), int(Nselect)))
    
  
    








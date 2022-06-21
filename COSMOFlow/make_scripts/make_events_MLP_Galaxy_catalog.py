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
import h5py 
np.random.seed(100)

from MultiLayerPerceptron.validate import run_on_dataset
from MultiLayerPerceptron.nn.model_creation import load_mlp
device = 'cpu'
model_name = 'SNR_approxiamator_sky'
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


zmax = 2

type_of_data = 'testing'
SNRth = 8
mth = 20


#grid of z and M
M_grid =  np.linspace(-23,-15,25)
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



def cdf_M(H0):
    para_dict ={'phi': 1.6*(10**-2)*(H0/100)**(3), 'alpha': -1.07, 'Mc': -20.47 + 5*np.log10(H0/100)}
    cdf = np.zeros(len(M_grid))
    for i in range(len(M_grid)):
        cdf[i] = quad(lambda M: priors.p_M(M, H0, para_dict),  M_grid [0], M_grid [i])[0]
    return cdf/np.max(cdf)    

def sample_M_from_cdf(cdf, N):
    t = rn.random(N)
    samples = np.interp(t,cdf,M_grid)
    return samples


if type_of_data == 'training':
    path_data = parentdir + r"/data_gwcosmo/galaxy_catalog/training_data_from_MLP/"
    N = 100000
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
    x_inds = [0,1, 2, 3, 4]
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
observed_RA = []
observed_dec = []
observed_m = []


# inx_in = np.where(H0_samples > 0)
H0 = H0_samples
while True:
    
    
    
    n = len(H0)
    
    M = [] 
    for cdf in cdfs:
        M.append(sample_M_from_cdf(cdf, 1))
        
    M = np.array(M)    
    
    
    #draw redshifts
    z = draw_cumulative_z(n, pz_int)
    
    #Draw RA dec 
    RA,dec = draw_RA_Dec(n)
    


    #compute distance and apparent magnitudes
    dl = cosmology.fast_z_to_dl_v2(np.array(z),np.array(H0))
    
    #Make sure all are arrays
    z = np.array(z)
    dl = np.array(dl)
    RA = np.array(RA)
    dec = np.array(dec)

    
    app_samples = cosmology.app_mag(M.flatten(),dl.flatten())

    inx_in_gal = np.where(np.array(app_samples) <= mth )[0]
    inx_out_gal = np.where(np.array(app_samples) > mth )[0]
    if len(inx_in_gal) > 0:
        #Add luminosity weights in the future 
        #Random choice glaaxy id with weights 
        gal_id = np.random.choice(np.linspace(0,len(catalog)-1, len(catalog)), size = len(inx_in_gal), p = weights/np.sum(weights))
        gal_selected = catalog.iloc[gal_id,:]

        RA_gal = np.array(gal_selected.ra)
        dec_gal = np.array(gal_selected.dec)
        z_true_gal = np.array(gal_selected.z)
        sigmaz_gal = np.array(gal_selected.sigmaz)
        z_obs_gal = np.random.uniform(z_true_gal, sigmaz_gal)
        m_obs_gal = np.array(gal_selected.m_B)

        dl_gal = cosmology.fast_z_to_dl_v2(np.array(z_obs_gal),np.array(H0[inx_in_gal]))

        #Switch z values in z array with zgal and dgal
        z[inx_in_gal] = z_obs_gal
        dl[inx_in_gal] = dl_gal 
        RA[inx_in_gal] = RA_gal
        dec[inx_in_gal] = dec_gal
        app_samples[inx_in_gal] = m_obs_gal
        
        
    

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

    data_dict = {'dl': dl, 'm1z':m1z, 'm2z':m2z, 'RA':RA, 'dec':dec}
    GW_data = pd.DataFrame(data_dict)


    snrs  = SNR_from_MLP(GW_data)
    GW_data['snr_true'] = snrs

    
    
    snrs_obs = np.sqrt((ncx2.rvs(4, snrs**2, size=n, loc = 0, scale = 1)))
    GW_data['snr'] = snrs_obs
    inx_out = np.where((GW_data.snr_true != 0 ) & (GW_data.snr >= SNRth) & (GW_data.snr < 150) & (GW_data.dl < 25000))

    
  
    
    observed_snr.append(snrs_obs[inx_out])
    observed_H0.append(H0[inx_out])
    observed_dl.append(dl[inx_out])
    observed_m1z.append(m1z[inx_out])
    observed_m2z.append(m2z[inx_out])
    observed_RA.append(RA[inx_out])
    observed_dec.append(dec[inx_out])
    
    H0 = np.delete(H0, inx_out)
    cdfs = np.delete(cdfs, inx_out, axis = 0 )


    
    
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
observed_RA = np.concatenate(observed_RA )
observed_dec = np.concatenate(observed_dec )




output_df = pd.DataFrame({'snr': observed_snr, 'H0': observed_H0, 'dl': observed_dl,
                          'm1': observed_m1z, 'm2': observed_m2z, 'RA': observed_RA, 'dec':observed_dec})
output_df.to_csv(path_data+'_data_{}_N_SNR_8.csv'.format(int(N)))



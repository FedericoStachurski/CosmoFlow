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
#print(parentdir)
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
import argparse
from poplar.nn.networks import LinearModel, load_model
from poplar.nn.rescaling import ZScoreRescaler
import torch
from tqdm import tqdm 
from scipy.special import gamma, gammaincc



#pass arguments 
#Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-Name", "--Name_file", required=True,
   help="Name of data")
ap.add_argument("-in_out", "--in_out", required=True,
   help="Make data with catalog or without (1 is in_out 0 is empty)", default = 1)
ap.add_argument("-batch", "--batch_number", required=True,
   help="batch number of the data", default = 1)
ap.add_argument("-type", "--type_data", required=True,
   help="Type of data? options [training, testing] ", default = 1)

ap.add_argument("-mass_distribution", "--mass_distribution", required=False,
   help="Mass distribution options [uniform, PowerLaw+Peak, PowerLaw]", default = 'PowerLaw+Peak')
ap.add_argument("-zmax", "--zmax", required=False,
   help="zmax", default = 1.5)
ap.add_argument("-zmin", "--zmin", required=False,
   help="zmin", default = 0.0001)
ap.add_argument("-H0max", "--H0max", required=False,
   help="H0 top boundary", default = 140)
ap.add_argument("-H0min", "--H0min", required=False,
   help="H0 bottom boundary", default = 20)
ap.add_argument("-SNRth", "--SNRth", required=False,
   help="SNR threshold", default = 11)
ap.add_argument("-band", "--magnitude_band", required=False,
   help="Magnitude band", default = 'K')
ap.add_argument("-n_detectors", "--n_detectors", required=False,
   help="Number of detectors to be used in the DoF in the Chi-squared distributions", default = 3)

ap.add_argument("-N", "--N", required=True,
   help="How many samples per batch", default = 100_000)
ap.add_argument("-Nselect", "--Nselect", required=False,
   help="Nselect per iteration", default = 5)
ap.add_argument("-threads", "--threads", required=False,
   help="threads", default = 10)
ap.add_argument("-device", "--device", required=False,
   help="device? [cpu, cuda]", default = 'cpu')
ap.add_argument("-seed", "--seed", required=False,
   help="Random seed", default = 1234)
ap.add_argument("-H0", "--H0", required=False,
   help="Hubble constant value for testing", default = 70)


args = vars(ap.parse_args())

Name = str(args['Name_file'])
in_out = str(args['in_out'])
batch = int(args['batch_number'])
type_of_data = str(args['type_data'])


mass_distribution = str(args['mass_distribution'])
zmax = float(args['zmax'])
zmin = float(args['zmin'])
Hmax = float(args['H0max'])
Hmin = float(args['H0min'])

SNRth = float(args['SNRth'])
mag_band = str(args['magnitude_band'])

N = int(args['N'])
Nselect = int(args['Nselect'])
threads = int(args['threads'])
n_det = int(args['n_detectors'])
device = str(args['device'])
seed = int(args['seed'])

H0_testing = float(args['H0'])



print()
print('Name model = {}'.format(Name))
print('in_out = {}'.format(in_out))
print('batch = {}'.format(batch))
print('type_of_data = {}'.format(type_of_data))
print('mass_distribution = {}'.format(mass_distribution))
print('zmax = {}'.format(zmax))
print('zmin = {}'.format(zmin))
print('SNRth = {}'.format(SNRth))
print('mag_band = {}'.format(mag_band))


if type_of_data == 'training':
    print('H0 = [{},{}]'.format(Hmin, Hmax))
else:    
    print('H0 = {}'.format(H0_testing))


print('N = {}'.format(N))
print('Nselect = {}'.format(Nselect))
print('threads= {}'.format(threads))
print('device= {}'.format(device))
print()

# cosmoflow/COSMOFlow/make_scripts/models/MLP_models/test_model_combined_O2
# cosmoflow/COSMOFlow/make_scripts/models/new_model/v1_SNR_v1O3
device = device
path_models = '/data/wiay/federico/PhD/cosmoflow/COSMOFlow/make_scripts/'
# print(os.getcwd())
# model_v1 = load_model(path_models+'models/new_model/v1_SNR_v1O3/model.pth', device = device)
model = load_model(path_models+'models/MLP_models/SNR_approximator_combined_O1_combined_O1/model.pth', device = device)
#model_h1 = load_model(path_models+'models/new_model/l1_SNR_v1O1/model.pth', device = device)





zmax = zmax
zmin = zmin
NSIDE = 32
type_of_data = type_of_data


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

in_out = str2bool(in_out)
SNRth = SNRth
Nselect = Nselect
N = N
distributions = {'mass':mass_distribution}
threads = threads
band = mag_band



np.random.seed(seed)

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


if band == 'Bj': 
    LF_para = {'Mabs_min': -22.00 , 'Mabs_max': -16.50, 'alpha': -1.21, 'Mstar': -19.66}
elif band == 'K':
    LF_para = {'Mabs_min': -27.00 , 'Mabs_max': -19, 'alpha': -1.09, 'Mstar': -23.39}
else: 
    raise ValueError(band + 'band is not implemented')


def M_grid_H0(low,high, H0):
    return np.linspace(high  + 5*np.log10(H0/100),low  + 5*np.log10(H0/100), 100)


def LF_weight_L(M,H0, phi_star =1.16*10**(-2), alpha = LF_para['alpha'], Mstar =LF_para['Mstar']):
    Mstar = Mstar + 5*np.log10(H0/100)
    phi_star = phi_star * (H0/100)**(3)
    return phi_star * 0.4*np.log(10)*10**(-0.4*(M - Mstar)*(alpha + 2))*np.exp(-10**(-0.4*(M - Mstar)))


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


def mag2lum(M):
    return 10**(M/(-2.5))


map_mth = np.loadtxt('/data/wiay/federico/PhD/cosmoflow/COSMOFlow/magnitude_threshold_maps/NSIDE_32_mth_map_GLADE_{}.txt'.format(band))
inx_0 = np.where(map_mth == 0.0 )[0]
map_mth[inx_0] = -np.inf


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
def Ngal_loaded_pix(pix):
    loaded_pix = catalog_pixelated[pix]
    loaded_pixel = loaded_pix[['z','RA','dec', 'sigmaz', 'm'+band]]
    loaded_pixel = loaded_pixel.dropna()
    loaded_pixel = loaded_pixel[loaded_pixel.z <= zmax]
    loaded_pixel = loaded_pixel[loaded_pixel.z >= zmin]
    Ngal = len(loaded_pixel)
    if Ngal <= 10:
        Ngal = 0 
    return Ngal

def load_pixel(pix):
    loaded_pix = catalog_pixelated[pix]
    loaded_pixel = loaded_pix[['z','RA','dec', 'sigmaz', 'm'+band]]
    loaded_pixel = loaded_pixel.dropna()
    loaded_pixel = loaded_pixel[loaded_pixel.z <= zmax]
    loaded_pixel = loaded_pixel[loaded_pixel.z >= zmin]
    return loaded_pix, len(loaded_pix)


def select_gal_from_pix(pixels_H0s):
    pixel, H0 = pixels_H0s
    loaded_pixel, Ngalpix = load_pixel(int(pixel))
    loaded_pixel = loaded_pixel[['z','RA','dec', 'sigmaz', 'm'+'K']]
    loaded_pixel = loaded_pixel[loaded_pixel.z <= zmax]
    loaded_pixel = loaded_pixel[loaded_pixel.z >= zmin]
    loaded_pixel['RA'] = np.deg2rad(loaded_pixel['RA'])
    loaded_pixel['dec'] = np.deg2rad(loaded_pixel['dec'])
    loaded_pixel = loaded_pixel.dropna()
    Ngalpix = len(loaded_pixel)
#     if loaded_pixel.empty is False:
    z_gal_selected = loaded_pixel.z 
    repeated_H0_in_pix = np.ones(Ngalpix)*H0
#     print(pixel)
#     print(len(repeated_H0_in_pix), len(z_gal_selected))
    dl_galaxies = cosmology.fast_z_to_dl_v2(np.array(z_gal_selected).flatten(),np.array(repeated_H0_in_pix).flatten())
    #get luminsoity
    absolute_mag = cosmology.abs_M(loaded_pixel['m'+band],dl_galaxies)

    luminosities = mag2lum(absolute_mag)
    weights_gal = luminosities * Madau_factor(z_gal_selected)
    weights_gal /= np.sum(weights_gal)
    gal_id = np.random.choice(np.arange(Ngalpix), size = 1, p = weights_gal)
    return loaded_pixel.iloc[gal_id,:]
#     else:
#         empty_dict = {'z':np.nan,'RA':np.nan,'dec':np.nan, 'sigmaz':np.nan, 'm'+band: np.nan}
#         return pd.DataFrame(empty_dict)
    




def h_samples_alpha(N, alpha, hmin=20, hmax=140):
    uniform_samples = np.random.uniform(0,1, size = N)
    return (uniform_samples*(hmax**(alpha)) + (1 - uniform_samples)*hmin**(alpha))**(1/alpha)



def cdf_LF(M, H0,  a = -1.09 , Mstar = -23.39, Mtop = -27, Mbottom = -19):
    Mstar = Mstar + 5*np.log10(H0/100)
    Mtop = Mtop  + 5*np.log10(H0/100)
    Mbottom = Mbottom  + 5*np.log10(H0/100)
    L_Lstar = np.power(10, -0.4*( M - Mstar ))
    Lmin_Lstar = np.power(10, -0.4*( Mbottom - Mstar ))
    Lmax_Lstar = np.power(10, -0.4*( Mtop - Mstar ))
    
    result = np.array((gammaincc(a+2, L_Lstar) - gammaincc(a+2, Lmax_Lstar)), dtype= float)
    return result 


def sample_M_from_cdf(cdf,H0,M_grid, N = 1):
    M_grid = M_grid_H0(LF_para['Mabs_min'], LF_para['Mabs_max'], H0)
    t = rn.random(N)
    samples = np.interp(t,cdf,M_grid)
    return samples



if type_of_data == 'training':
    if in_out is True: 
        path_data = parentdir + r"/data_gwcosmo/galaxy_catalog/training_data_from_MLP/"
    else:
        path_data = parentdir + r"/data_gwcosmo/empty_catalog/training_data_from_MLP/"
    H0_samples = h_samples_alpha(N,1, hmin = Hmin, hmax = Hmax)#np.random.uniform(Hmin,Hmax,N)
    if in_out is True:
        M_grid = M_grid_H0(LF_para['Mabs_min'], LF_para['Mabs_max'], 100)
        cdf_M = cdf_LF(M_grid[::-1], 100)
        
    else: 
        cdfs_H0 =  np.random.uniform(0,1,N)

if type_of_data == 'testing': 
    if in_out is True: 
        path_data = parentdir + r"/data_gwcosmo/galaxy_catalog/testing_data_from_MLP/"
    else:
        path_data = parentdir + r"/data_gwcosmo/empty_catalog/testing_data_from_MLP/"

    H0_samples = H0_testing*np.ones(N)
    R_nums = np.random.uniform(0,1, size = N)
    if in_out is True:
        M_grid = M_grid_H0(LF_para['Mabs_min'], LF_para['Mabs_max'], 100)
        cdf_M = cdf_LF(M_grid[::-1], 100)

    
    
#functions    
def ha_from_ra_gps(ra, gps):
    return (gps/(3600*24))%(86164.0905/(3600*24))*2*np.pi - ra


def prep_data_for_MLP(df):
    data_testing = df[['mass_1','mass_2',
                        'ra','dec',
                       'theta_jn',
                       'psi','geocent_time']]
    ha_testing = ha_from_ra_gps(np.array(data_testing.ra), np.array(data_testing.geocent_time))
    df['ha'] = ha_testing
    data_testing = df[['mass_1','mass_2', 'theta_jn', 'ha', 'dec', 'psi' ]]
    xdata_testing = torch.as_tensor(data_testing.to_numpy(), device=device).float()
    return xdata_testing


H0 = H0_samples
N_missed = N
list_data = []
missed_H0 = H0
# missed_cdfs = cdfs_H0
if type_of_data =='testing':
    missed_R = R_nums

    
    
# with tqdm(total=N, desc = 'Total events computed') as pbar:
while True:    
    st = time.perf_counter()
    start = time.time()
    n = len(missed_H0)
    select = int(Nselect)#*(N/N_missed)**1)
    nxN = int(n*select)
    repeated_H0 = np.repeat(missed_H0, select)
    
    inx_gal = np.zeros(nxN)
    #draw redshifts
    z = draw_cumulative_z(nxN)

    dl = cosmology.fast_z_to_dl_v2(np.array(z),np.array(repeated_H0))
    #Make sure all are arrays
    z = np.array(z)
    dl = np.array(dl)
    RA, dec = draw_RA_Dec(int(nxN))
    if in_out is True:
        M_abs = sample_M_from_cdf(cdf_M, 100, M_grid, N = nxN)
        M_abs = M_abs + 5*np.log10(repeated_H0/100) 
        app_samples = cosmology.app_mag(M_abs.flatten(),dl.flatten())


    if in_out is True:
        #Magnitude threshold map
        mth_list = np.array([mth_from_RAdec(NSIDE, RA, dec, map_mth)]).flatten()
        pix_list = np.array([pix_from_RAdec(NSIDE, RA, dec)]).flatten()
#         inx_in_gal = np.where((app_samples < np.inf) == True)[0] 
#         pix_list_in = np.array(pix_list[inx_in_gal])
        with multiprocessing.Pool(threads) as p:
            Ngal_in_pix = list(p.imap(Ngal_loaded_pix,pix_list))
            
        inx_non_zero = np.where(np.array(Ngal_in_pix) > 0)[0]
#         print(inx_non_zero, np.array(Ngal_in_pix)[inx_non_zero])
        pix_list = np.array(pix_list[inx_non_zero])
        H0_in_list = np.array(repeated_H0[inx_non_zero])
#         print(pix_list, len(H0_in_list))
        pixel_H0 = np.array([pix_list, H0_in_list]).T
        with multiprocessing.Pool(threads) as p:
            selected_cat_pixels = list(p.imap(select_gal_from_pix,pixel_H0))   
#         print(selected_cat_pixels)
        gal_selected = pd.concat(selected_cat_pixels)
        inx_in_gal = inx_non_zero
        
    
        if len(inx_in_gal) > 0:

            #Add luminosity weights in the future 
            #Random choice glaaxy id with weights 

#             print(len(selected_cat_pixels))
            RA_gal = np.array(gal_selected.RA)
            dec_gal = np.array(gal_selected.dec)
            z_true_gal = np.array(gal_selected.z)
            sigmaz_gal = np.array(gal_selected.sigmaz)
            #z_obs_gal = np.random.normal(z_true_gal, sigmaz_gal)
            a, b = (zmin - z_true_gal) / sigmaz_gal, (zmax - z_true_gal) / sigmaz_gal
            z_obs_gal = truncnorm.rvs(a, b, loc=z_true_gal, scale=abs(sigmaz_gal), size=len(z_true_gal))
            m_obs_gal = np.array(gal_selected['m'+band])
#             print(len(z_obs_gal), len(H0_in_list))
            dl_gal = cosmology.fast_z_to_dl_v2(np.array(z_obs_gal),np.array(H0_in_list))

            #Switch z values in z array with zgal and dgal
            z[inx_in_gal] = z_obs_gal
            dl[inx_in_gal] = dl_gal 
            RA[inx_in_gal] = RA_gal
            dec[inx_in_gal] = dec_gal
            app_samples[inx_in_gal] = m_obs_gal
            inx_gal[inx_in_gal] = 1
            end = time.time()

    if type_of_data == 'testing':
        repeated_Rnums = np.repeat(missed_R, select) 
    

    start = time.time()
    _, m1, m2, _, _, _, _, _, _, theta_jn, _, _, psi, _ , geo_time = gw_priors.draw_prior(nxN, distributions)
    end = time.time()
    m1z = m1*(1+z)
    m2z = m2*(1+z)


    data_dict = {'luminosity_distance':dl, 'mass_1':m1z, 'mass_2':m2z,'a1': 0, 'a2': 0,
                 'tilt1': 0, 'tilt2': 0,'ra':RA, 'dec':dec,
                 'theta_jn':theta_jn, 'phi_jl':0, 'phi_12':0, 'psi':psi , 'geocent_time':geo_time}  
#     data_dict = {'luminosity_distance':dl, 'mass_1':m1z, 'mass_2':m2z,
#                  'ra':RA, 'dec':dec,'theta_jn':theta_jn, 'psi':psi , 'geocent_time':geo_time}   

    GW_data = pd.DataFrame(data_dict)
    ### SNR calulcator using MLPs ###
    x_data_MLP  = prep_data_for_MLP(GW_data)
    ypred = model.run_on_dataset(x_data_MLP)
    snr_pred = ypred.cpu().numpy()/np.array(GW_data['luminosity_distance'])
    snrs_obs = np.sqrt((ncx2.rvs(2*n_det, snr_pred**2, size=nxN, loc = 0, scale = 1)))        
    et = time.perf_counter()

    GW_data['snr'] = snrs_obs 
    GW_data['inx'] = inx_gal
    inx_out = np.where((GW_data.snr >= SNRth) & (GW_data.inx == 1))[0]   
    GW_data['H0'] = repeated_H0
    if in_out is True:
        GW_data['app_mag'] = app_samples
    else:
        GW_data['app_mag'] = np.ones(len(repeated_H0))


    if type_of_data =='testing':
        GW_data['R'] = repeated_Rnums

    if Nselect > 1:
        inds_to_keep = []
        for k in range(n):
            try:
                inds_to_keep.append(inx_out[(k*int(select) < inx_out) & (inx_out < (k+1)*int(select))][0])
            except IndexError:
                pass
        if len(inds_to_keep) == 0:
            continue
    else: 
        inds_to_keep = inx_out
    
    out_data = GW_data.loc[np.array(inds_to_keep)]
    list_data.append(out_data)


    if type_of_data =='training':
        missed_H0 = np.setxor1d(out_data['H0'].to_numpy(),repeated_H0) 
        sys.stdout.write('\rH0 we missed: {} | Nselect = {} | nxN = {}'.format(len(missed_H0), Nselect,  nxN))
        N_missed = len(missed_H0)   
        if len(missed_H0) == 0 : 
            repeated_H0 = np.repeat(missed_H0, select)
            break

    elif type_of_data =='testing':
        missed_R = np.setxor1d(out_data['R'].to_numpy(),repeated_Rnums)

        temp = []
        for x in R_nums:
            temp.append(np.where(missed_R == x)[0])

        indices_R = np.concatenate(temp,axis=0) 
        missed_H0 = missed_H0[indices_R]

        
        sys.stdout.write('\rH0 we missed: {} | Nselect = {} '.format(len(missed_H0), nxN))
        N_missed = len(missed_H0)
        if N_missed == 0 : 
            repeated_H0 = np.repeat(missed_H0, select)
            break
        

print('\nFINISHED Sampling events')  
GW_data = pd.concat(list_data)    
output_df = GW_data[['snr', 'H0', 'luminosity_distance', 'mass_1', 'mass_2', 'ra', 'dec',
                     'a1', 'a2', 'tilt1', 'tilt2', 'theta_jn',
                     'phi_jl', 'phi_12', 'psi','geocent_time', 'app_mag', 'inx']]
output_df.to_csv(path_data+'name_{}_catalog_{}_band_{}_batch_{}_N_{}_SNR_{}_Nelect_{}__Full_para_v1.csv'.format(Name, in_out,band, int(batch), int(N), int(SNRth), int(Nselect)))
    
  
    


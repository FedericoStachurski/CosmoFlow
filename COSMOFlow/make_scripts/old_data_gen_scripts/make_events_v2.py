#Author: Federico Stachurski 
#Procedure: generate data frame of GW observables 
#Input: -N number of evetns, -batch_size how many batches to save from N, -SNRth threshold, -zmax redhsift maximum -mth magnitude threshold 
#Output: data frame of galaxies
#Date: 07/08/2021

# python3 make_events.py -Name test_hot -batch 1 -in_out 1 -type training -mass_distribution PowerLaw+Peak -zmax 1.8 -zmin 0.0001 -H0 20 180 -gamma 0.0 12.0 -k 2.86 -zp 2.47 -mmax 50 200 -mmin 4.98 -alpha 3.78 -mu_g 20 50 -sigma_g 3.88 -lambda_peak 0.03 -beta 0.81 -delta_m 4.8  -SNRth 11.0 -SNRth_single 0.0 -band K -run O3 -detectors H1 L1 V1 -N 5 -Nselect 2 -device cuda:0 -threads 10 -seed 116

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

from cosmology_functions import priors, cosmology, utilities
from cosmology_functions.z_parameters_dist import RedshiftGW_fast_z_para
from cosmology_functions.schechter_functions import Schechter_function

from gw_functions import gw_priors_v2
from gw_functions import gw_SNR
from gw_functions.mass_prior_multi_para_test import MassPrior

import warnings
warnings.filterwarnings("ignore")

import time
from scipy.stats import chi2
import numpy.random as rn
from scipy.stats import ncx2
from scipy.interpolate import splrep, splev
from scipy.stats import truncnorm
# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
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
import cupy as cp
xp = cp



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

# Cosmology Setup
ap.add_argument("-H0", "--H0", nargs='+', required=True,
   help="Hubble values: OPTIONS [min, max] or value", default = 70)
ap.add_argument("-gamma", "--gamma", nargs='+', required=True,
   help="gamma values: OPTIONS [min, max] or value", default = 70)
ap.add_argument("-k", "--k", nargs='+', required=True,
   help="kappa values: OPTIONS [min, max] or value", default = 70)
ap.add_argument("-zp", "--zp", nargs='+', required=True,
   help="zp values: OPTIONS [min, max] or value", default = 70)
ap.add_argument("-alpha", "--alpha", nargs='+', required=True,
   help="alpha values: OPTIONS [min, max] or value", default = 70)
ap.add_argument("-mmax", "--mmax", nargs='+', required=True,
   help="mmax values: OPTIONS [min, max] or value", default = 70)
ap.add_argument("-mmin", "--mmin", nargs='+', required=True,
   help="mmin values: OPTIONS [min, max] or value", default = 70)
ap.add_argument("-mu_g", "--mu_g", nargs='+', required=True,
   help="mu_g values: OPTIONS [min, max] or value", default = 70)
ap.add_argument("-sigma_g", "--sigma_g", nargs='+', required=True,
   help="sigma_g values: OPTIONS [min, max] or value", default = 70)
ap.add_argument("-lambda_peak", "--lambda_peak", nargs='+', required=True,
   help="lambda_peak values: OPTIONS [min, max] or value", default = 70)
ap.add_argument("-beta", "--beta", nargs='+', required=True,
   help="beta values: OPTIONS [min, max] or value", default = 70)
ap.add_argument("-delta_m", "--delta_m", nargs='+', required=True,
   help="detla_m values: OPTIONS [min, max] or value", default = 70)



#Detector and EM catalog setup

ap.add_argument("-SNRth", "--SNRth", required=False,
   help="SNR threshold", default = 11)
ap.add_argument("-SNRth_single", "--SNRth_single", required=False,
   help="SNR threshold", default = 0)
ap.add_argument("-band", "--magnitude_band", required=False,
   help="Magnitude band", default = 'K')
ap.add_argument("-run", "--run", required=False,
   help="Detector run [O1, O2, O3]", default = 'O1')
ap.add_argument("-detectors", "--detectors", nargs='+', required=True,
   help="make data from detector: OPTIONS [H1, L1, V1]", default = 'H1')
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
ap.add_argument("-fast_zmax", "--fast_zmax", required=False,
   help="Makes the code faster by reducing zmax for specific H0", default = 1)
ap.add_argument("-save_timer", "--save_timer", required=False,
   help="If 1 save agnostic data of data generation, else not", default = 0)
ap.add_argument("-approximator", "--approximator", required=False,
   help="waveform approxiamtor ", default = 'IMRPhenomXPHM')
ap.add_argument("-name_pop", "--name_pop", required=False,
   help="type of population", default = 'BBH-powerlaw-gaussian')

args = vars(ap.parse_args())
Name = str(args['Name_file'])
in_out = str(args['in_out'])
batch = int(args['batch_number'])
type_of_data = str(args['type_data'])
mass_distribution = str(args['mass_distribution'])
name_pop = str(args['name_pop'])
zmax = float(args['zmax'])
zmin = float(args['zmin'])


H0 = list(map(float,args['H0']))
gamma = list(map(float,args['gamma']))
k = list(map(float,args['k']))
zp = list(map(float,args['zp']))
mmax = list(map(float,args['mmax']))
mmin = list(map(float,args['mmin']))
alpha = list(map(float,args['alpha']))
mu_g = list(map(float,args['mu_g']))
sigma_g = list(map(float,args['sigma_g']))
lambda_peak = list(map(float,args['lambda_peak']))
beta = list(map(float,args['beta']))
delta_m = list(map(float,args['delta_m']))


SNRth = float(args['SNRth'])
SNRth_single = float(args['SNRth_single'])
mag_band = str(args['magnitude_band'])
N = int(args['N'])
Nselect = int(args['Nselect'])
threads = int(args['threads'])
run = str(args['run'])
detectors = args['detectors']
n_det = len(detectors)
# n_det = int(args['n_detectors'])
device = str(args['device'])
fast_zmax = int(args['fast_zmax'])
seed = int(args['seed'])
save_timer = int(args['save_timer'])
approximator = str(args['approximator'])

print()
print('Name model = {}'.format(Name))
print('in_out = {}'.format(in_out))
print('batch = {}'.format(batch))
print('type_of_data = {}'.format(type_of_data))
print('mass_distribution = {}'.format(mass_distribution))
print('population type = {}'.format(name_pop))
print('zmax = {}'.format(zmax))
print('zmin = {}'.format(zmin))
print('SNRth_combined_network = {}'.format(SNRth))
print('SNRth_single_detector = {}'.format(SNRth_single))
print('mag_band = {}'.format(mag_band))
print('n_detectors = {}'.format(n_det))
print('detectors = {}'.format(detectors))
print('approximator = {}'.format(approximator))
print('run = {}'.format(run))
print('H0 = {}'.format(H0))
print('Gamma = {}'.format(gamma))
print('k = {}'.format(k))
print('zp = {}'.format(zp))
print('alpha = {}'.format(alpha))
print('beta = {}'.format(beta))
print('mmax = {}'.format(mmax))
print('mmin = {}'.format(mmin))
print('mu_g = {}'.format(mu_g))
print('sigma_g = {}'.format(sigma_g))
print('lambda_peak = {}'.format(lambda_peak))
print('delta_m = {}'.format(delta_m))
print('N = {}'.format(N))
print('Nselect = {}'.format(Nselect))
print('threads = {}'.format(threads))
print('device = {}'.format(device))
print('fast_zmax = {}'.format(fast_zmax))
print('save_timer= {}'.format(save_timer))
print('seed = {}'.format(seed))
print()



def decide_either_uniform_or_ones(value,N):
    if len(value) > 1:
        return np.random.uniform(value[0],value[1],N)
    elif len(value) == 1:
        return np.ones(N)*value
    
    
#Sample from flat pripors (or fixed values) Cosmological and population parameters 
H0 = decide_either_uniform_or_ones(H0,N) ; Om0 = 0.3 ; w0 = -1.0
gamma = decide_either_uniform_or_ones(gamma,N) ; k = decide_either_uniform_or_ones(k,N) ; zp = decide_either_uniform_or_ones(zp,N)
alpha = decide_either_uniform_or_ones(alpha,N) ; beta = decide_either_uniform_or_ones(beta,N); mmax = decide_either_uniform_or_ones(mmax,N)
mmin = decide_either_uniform_or_ones(mmin,N) ; mu_g = decide_either_uniform_or_ones(mu_g,N) ; sigma_g = decide_either_uniform_or_ones(sigma_g,N)
lambda_peak = decide_either_uniform_or_ones(lambda_peak,N) ; delta_m = decide_either_uniform_or_ones(delta_m,N)


indicies_detectors = [] #Check which detectors to use
if 'H1' in detectors: 
    indicies_detectors.append(0)
if 'L1' in detectors:
    indicies_detectors.append(1)
if 'V1' in detectors:
    indicies_detectors.append(2)

if run == 'O1':
    model = load_model('models/MLP_models/SNR_MLP_TOTAL_v2_{}_{}_H1_L1/model.pth'.format(approximator, run), device = device) #load MLP model 
    print('SNR approxiamtor = SNR_approximator_{}_{}_H1_L1'.format(approximator, run))
    
else: 
    model = load_model('models/MLP_models/SNR_MLP_TOTAL_v2_{}_{}_H1_L1_V1/model.pth'.format(approximator, run), device = device) #load MLP model 
    print('SNR approxiamtor = SNR_approximator_{}_{}_H1_L1_V1'.format(approximator, run))


in_out = utilities.str2bool(in_out) #cehck if with catalogue or no catalogue
band = mag_band #magnitude band to use 
np.random.seed(seed) # set random seed 

#Load  pixelated galaxy catalog
NSIDE = 32  #Define NSIDE for healpix map
Npix = hp.nside2npix(NSIDE)


# H0 = np.sort(H0) ; gamma = np.sort(gamma) ; k = np.sort(k) ; zp = np.sort(zp)

#define cosmological parameters of GWs
cosmological_parameters = {'gamma': gamma, 'k': k, 'zp': zp, 'lam': 0, 'Om0':Om0, 'w0': w0, 'H0': H0}
### Initiate redshift and mass classes 
z_class = RedshiftGW_fast_z_para(cosmological_parameters, zmin = zmin , zmax = zmax, run = run, SNRth = SNRth)#initiate zmax calss for zmax = f(H0, SNRth) #Hcekc if option is used 
population_parameters = {'beta': xp.array(beta), 'alpha': xp.array(alpha), 'mmin': xp.array(mmin) ,'mmax': xp.array(mmax), 
                             'mu_g': xp.array(mu_g), 'sigma_g': xp.array(sigma_g), 
                             'lambda_peak': xp.array(lambda_peak),'delta_m': xp.array(delta_m), 'name': name_pop}
# print(np.shape(population_parameters['delta_m'].get()), np.shape(population_parameters['mmin'].get()))
mass_class_temp = MassPrior(population_parameters, mgrid = 250) #initiate mass prior class, p(m1,m2)


if in_out is True: #check if using a catalog
    sch_fun = Schechter_function(band) #initiate luminosity functions class 
    
    def load_cat_by_pix(pix): #load pixelated catalog 
        loaded_pix = pd.read_csv('/data/wiay/federico/PhD/cosmoflow_review/COSMOFlow/pixelated_catalogs/GLADE+_pix/pixel_{}'.format(pix)) #Include NSIDE in the name of folders 
        return loaded_pix
    
    def load_pixel(pix): #load pixel from catalog
        loaded_pix = catalog_pixelated[pix]
        return loaded_pix, len(loaded_pix)

    with multiprocessing.Pool(threads) as p: #begin multiprocesseing for loading the catalog
        catalog_pixelated = list(tqdm(p.imap(load_cat_by_pix,np.arange(Npix)), total = Npix, desc = 'GLADE+ catalog, NSIDE = {}'.format(NSIDE)))

    #load mth map for specific filter band 
    map_mth = np.loadtxt('/data/wiay/federico/PhD/cosmoflow_review/COSMOFlow/magnitude_threshold_maps/NSIDE_32_mth_map_GLADE_{}.txt'.format(band))
    inx_0 = np.where(map_mth == 0.0 )[0] #if mag threshold is zero, set it to -inf 
    map_mth[inx_0] = -np.inf
        

def select_gal_from_pix(pixels_H0_gamma_para): 
    # "Selects galaxies from pixel using pixel index and associated H0 to pixel"
    # "Input: tuple(pixel_inx,H0); 
    # "Returns: dataframe of pixel id (z, ra, dec...)" 
    
    pixel, H0, gamma, k, zp = pixels_H0_gamma_para
    loaded_pixel, Ngalpix = load_pixel(int(pixel))
    loaded_pixel = loaded_pixel[['z','RA','dec', 'sigmaz', 'm'+band]] #load pixel 
    loaded_pixel = loaded_pixel.dropna() # drop any Nan values 
    
    temporary_zmax = z_class.zmax_H0(H0, SNRth) #for compelteness use the zmax at the given H0 
    
    loaded_pixel = loaded_pixel[loaded_pixel.z <= temporary_zmax] #check if redshift is less than zmax at H0 value
    loaded_pixel = loaded_pixel[loaded_pixel.z >= zmin] #check if z is greater than zmin 
    loaded_pixel['RA'] = np.deg2rad(loaded_pixel['RA']) #convert RA and dec into radians 
    loaded_pixel['dec'] = np.deg2rad(loaded_pixel['dec'])

    Ngalpix = len(loaded_pixel) #get number of galaxies in pixel 
    
    if loaded_pixel.empty is False: #if there are galaxies in the pixel
        z_gal_selected = loaded_pixel.z #get redshift 
        repeated_H0_in_pix = np.ones(Ngalpix)*H0 #for that specific pixel, make vector of H0s used for the specific pixel 
        dl_galaxies = cosmology.fast_z_to_dl_v2(np.array(z_gal_selected).flatten(),np.array(repeated_H0_in_pix).flatten()) #compute distances of galaxies using redshift and H0 
        
        #get luminsoity
        absolute_mag = cosmology.abs_M(loaded_pixel['m'+band],dl_galaxies)
        luminosities =  cosmology.mag2lum(absolute_mag)
        
        #weights = L * madau(z) * (1/(1+z))
        weights_gal = luminosities * z_class.Madau_factor(z_gal_selected, zp, gamma, k) * z_class.time_z(z_gal_selected)
        weights_gal /= np.sum(weights_gal) # check weights sum to 1
        gal_id = np.random.choice(np.arange(Ngalpix), size = 1, p = weights_gal) #random choice of galaxy in the pixel 
        return loaded_pixel.iloc[gal_id,:]
    
    else: #if no galaxies in pixel, return None 
        return None 

    



if type_of_data == 'training':
    #if making training data, define paths where to store the data, and sample H0 
    if in_out is True: 
        path_data = parentdir + r"/data_cosmoflow/galaxy_catalog/training_data_from_MLP/"
    else:
        path_data = parentdir + r"/data_cosmoflow/empty_catalog/training_data_from_MLP/"
        
    cdfs_zmax = z_class.make_cdfs()
    cdfs_m1 = mass_class_temp.make_cdfs()
    
    
if type_of_data == 'testing': #
    if in_out is True: 
        path_data = parentdir + r"/data_cosmoflow/galaxy_catalog/testing_data_from_MLP/"
    else:
        path_data = parentdir + r"/data_cosmoflow/empty_catalog/testing_data_from_MLP/"

    R_nums = np.random.uniform(0,1, size = N) #this is a random number which is used to keep track of which H0 is being used at any given moment 
    cdfs_zmax = z_class.make_cdfs()
    cdfs_m1 = mass_class_temp.make_cdfs()

    
#rename variables (should change this) #Add more comments explainign everything 
# H0 = H0_samples 
N_missed = N
list_data = []

missed_H0 = H0 ; missed_gamma = gamma ; missed_kappa = k ; missed_zp = zp
missed_alpha = alpha ; missed_beta = beta ; missed_mmax = mmax ; missed_mmin = mmin
missed_mu_g = mu_g ; missed_sigma_g = sigma_g ; missed_lambda = lambda_peak ; missed_delta_m = delta_m
missed_cdfs_zmax = cdfs_zmax
missed_cdfs_m1 = cdfs_m1


if type_of_data =='testing':
    missed_R = R_nums

counter = 0 #variables to stare for save counter 
counter_list = []
Nmissed_list = []
timer_list = [] #

#begin loop for generating data 
while True: 
    
    start = time.time() # start timer to check efficency in save counter 
    n = len(missed_H0) #check how many H0s are there to be detected
    select = int(Nselect*(N/N_missed)) #use this is the selct value, which increases the more H0s are detected
    nxN = int(n*select) #defin the number of samples we are going to sample nxN (n = H0s, N = sampels per H0s)
    
    if type_of_data == 'testing':
        repeated_Rnums = np.repeat(missed_R, select) 
    
    repeated_H0 = np.repeat(missed_H0, select) #repeat H0s for Nselect samples 
    repeated_gamma = np.repeat(missed_gamma, select) #repeat Gammas for Nselect samples 
    repeated_kappa = np.repeat(missed_kappa, select) #repeat Kappas for Nselect samples 
    repeated_zp = np.repeat(missed_zp, select) #repeat zps for Nselect samples 
    repeated_alpha = np.repeat(missed_alpha, select) #repeat alphas for Nselect samples 
    repeated_beta = np.repeat(missed_beta, select) #repeat betas for Nselect samples 
    repeated_mmax = np.repeat(missed_mmax, select) #repeat mmaxs for Nselect samples 
    repeated_mmin = np.repeat(missed_mmin, select) #repeat mmins for Nselect samples 
    repeated_mu_g = np.repeat(missed_mu_g, select) #repeat mus for Nselect samples 
    repeated_sigma_g = np.repeat(missed_sigma_g, select) #repeat sigmas for Nselect samples 
    repeated_lambda = np.repeat(missed_lambda, select) #repeat lambdas for Nselect samples 
    repeated_delta_m = np.repeat(missed_delta_m, select) #repeat delta_ms for Nselect samples 
    
    
    population_parameters = {'beta': xp.array(repeated_beta), 'alpha': xp.array(repeated_alpha), 'mmin': xp.array(repeated_mmin) ,'mmax': xp.array(repeated_mmax), 
                             'mu_g': xp.array(repeated_mu_g), 'sigma_g': xp.array(repeated_sigma_g), 
                             'lambda_peak': xp.array(repeated_lambda),'delta_m': xp.array(repeated_delta_m), 'name': name_pop}
    # print(np.shape(population_parameters['delta_m'].get()), np.shape(population_parameters['mmin'].get()))
    mass_class = MassPrior(population_parameters, mgrid = 100) #initiate mass prior class, p(m1,m2)
    
    
    inx_gal = np.zeros(nxN) #define galxy indecies 
    
    
    RA, dec = cosmology.draw_RA_Dec(nxN) #sample RA and dec 
    z = z_class.draw_z_zmax(select, missed_cdfs_zmax.T) #sampleredshift from zmax-H0 distributions 
    dl = cosmology.fast_z_to_dl_v2(np.array(z),np.array(repeated_H0)) #convert z-H0s into luminosity distances 
    
    #Make sure all are arrays
    z = np.array(z)
    dl = np.array(dl)
    
    #If using galaxy catalog 
    if in_out is True:
        M_abs = sch_fun.sample_M_from_cdf_weighted(100, N = nxN) #sample absolute magnitudes from H0 = 100
        M_abs = M_abs + 5*np.log10(repeated_H0/100) #shift absolute magnitudes by 5log10(H0/100) to conver them 
        app_samples = cosmology.app_mag(M_abs.flatten(),dl.flatten()) #compute apparent magnitudes 
        
        #Handle Magnitude threshold map
        mth_list = np.array([utilities.mth_from_RAdec(NSIDE, RA, dec, map_mth)]).flatten() #list of mths 
        pix_list = np.array([utilities.pix_from_RAdec(NSIDE, RA, dec)]).flatten() #list of pixels per mth
        inx_in_gal = np.where((app_samples < mth_list) == True)[0]  #check where theapp_mag is brighter than the mth (if yes, that is a galaxy in the catalog)

        if len(inx_in_gal) > 0: #if the nubmer of galaxies selected is greater than zero, start galaxy selection

            pix_list = np.array(pix_list[inx_in_gal]) #get list of pixels from where to get the galaxies 
            H0_in_list = np.array(repeated_H0[inx_in_gal]) #get the associated H0s from each pixel
            gamma_in_list = np.array(repeated_gamma[inx_in_gal]) #get the associated H0s from each pixel
            kappa_in_list = np.array(repeated_kappa[inx_in_gal]) #get the associated H0s from each pixel
            zp_in_list = np.array(repeated_zp[inx_in_gal]) #get the associated H0s from each pixel
            
            pixel_H0 = np.array([pix_list, H0_in_list, gamma_in_list, kappa_in_list, zp_in_list]).T #make an array of lists with pixel index and H0 to be used in the select galaxy function

            with multiprocessing.Pool(threads) as p: #multiprocess for galaxy pixel loading 
                selected_cat_pixels = list(p.imap(select_gal_from_pix,pixel_H0))
           
            if len(selected_cat_pixels) >= 1: #if we have selected more or one galaxy
                gal_selected = pd.concat(selected_cat_pixels) #get selected galaxy 
                RA_gal = np.array(gal_selected.RA) #get RA of galaxy 
                dec_gal = np.array(gal_selected.dec) #get dec of galaxy
                z_true_gal = np.array(gal_selected.z) #get redshift of galaxy 
                sigmaz_gal = np.array(gal_selected.sigmaz) #get redshift uncertainty 
                a, b = (zmin - z_true_gal) / sigmaz_gal, (zmax - z_true_gal) / sigmaz_gal #sample from truncated gaussian redshift using the uncertainty 
                z_obs_gal = truncnorm.rvs(a, b, loc=z_true_gal, scale=abs(sigmaz_gal), size=len(z_true_gal))
                m_obs_gal = np.array(gal_selected['m'+band]) #get the apparent magnitude 

                dl_gal = cosmology.fast_z_to_dl_v2(np.array(z_obs_gal),np.array(H0_in_list)) #compute the distance 
                #Switch z values in z array with zgal and dgal
                z[inx_in_gal] = z_obs_gal #switch galaxies from initial set with galaxies sampled fro mgalaxy catalog 
                dl[inx_in_gal] = dl_gal 
                RA[inx_in_gal] = RA_gal
                dec[inx_in_gal] = dec_gal
                app_samples[inx_in_gal] = m_obs_gal
                inx_gal[inx_in_gal] = 1 #set the index of that event 1, to identify it was fro mthe galaxy catalog 

                ###### NOTE: scatter weights beofre instead of here. 
    
    #sample priors on theta_jn, psi, geo_time
    #_, _, _, _, _, _, _, _, _, theta_jn, _, _, psi, _ , geo_time = gw_priors_v2.draw_prior(int(nxN))
    _, _, _, a1, a2, tilt1, tilt2, _, _, theta_jn, phi_jl, phi_12, psi, _, geo_time = gw_priors_v2.draw_prior(int(nxN))
    
    
    #Sample primary and secondary masses
    # m1, m2 = mass_class.PL_PEAK_GWCOSMO(nxN) #use GWCOSMO mass prior distribution #NOTE: Deifne better 
    samples_m1 = mass_class_temp.draw_m(select, missed_cdfs_m1)
    samples_m1[samples_m1 < population_parameters['mmin'].get()] = 0.0
    m2_array_vect = np.concatenate(xp.linspace(xp.zeros(len(samples_m1)), samples_m1, mass_class.mgrid, axis = 1))
    cdfs_m2 = mass_class.make_cdfs_m2(xp.array(samples_m1))
    samples_m2 = mass_class.draw_m(1, cdfs_m2,  m_array_long = m2_array_vect)
    
    samples_m2[samples_m2< population_parameters['mmin'].get()] = 0.0
    samples_m1[np.isnan(samples_m2)] = 0.0
    samples_m2[np.isnan(samples_m2)] = 0.0
    
    m1z = samples_m1*(1+z) #turn source masses into detector frame masses 
    m2z = samples_m2*(1+z)
    
    data_dict = {'luminosity_distance':dl, 'mass_1':m1z, 'mass_2':m2z,'a_1': a1, 'a_2': a2,
             'tilt_1': tilt1, 'tilt_2': tilt2,'ra':RA, 'dec':dec,
             'theta_jn':theta_jn, 'phi_jl':phi_jl, 'phi_12':phi_12, 'psi':psi , 'geocent_time':geo_time}  #define dictionary with relevant GW parameters 

    GW_data = pd.DataFrame(data_dict) #make GW data frame 
    
    ### SNR calulcator using MLP ###
    x_data_MLP  = utilities.prep_data_for_MLP_full(GW_data, device) #prep data to be passes to MLP 
    ypred = model.run_on_dataset(x_data_MLP.to(device)) #get the predicted y values (SNR * distance)
    snr_pred = ypred.cpu().numpy()/np.array(GW_data['luminosity_distance'])[:,None] #divide by distance to get SNR 

    temp_dict = {} 
    temp_snrs = []
    if 'H1' in detectors: 
        temp_dict['snr_h1'] = snr_pred[:,0]
        temp_snrs.append(snr_pred[:,0])
    if 'L1' in detectors: 
        temp_dict['snr_l1'] = snr_pred[:,1]
        temp_snrs.append(snr_pred[:,1])
    if 'V1' in detectors: 
        temp_dict['snr_v1'] = snr_pred[:,2]
        temp_snrs.append(snr_pred[:,2])
    
    network_snr_sq = np.sum((np.array(temp_snrs)**2).T, axis = 1) #get detector netwrok snr 
    snrs_obs = np.sqrt((ncx2.rvs(2*n_det, network_snr_sq, size=nxN, loc = 0, scale = 1))) #sample from non central chi squared with non centrality parameter SNR**2
    snrs_obs[samples_m2 == 0 ] = 0
    temp_dict['observed'] = snrs_obs   
    df_temp_snrs = pd.DataFrame(temp_dict)
    
    if SNRth_single > 0:
    # if SNRth_single is > 0, then check if individual GW event was detected in each individaul detector, only keep those 
        if 'H1' and 'L1' in detectors:
            bad_inx = np.where(((df_temp_snrs['snr_h1']>SNRth_single) & (df_temp_snrs['snr_l1']>SNRth_single) & (df_temp_snrs['observed']>SNRth)) == False)[0]
            df_temp_snrs.loc[bad_inx] = np.nan
            GW_data['H1'] = np.array(df_temp_snrs.snr_h1)
            GW_data['L1'] = np.array(df_temp_snrs.snr_l1)

        if 'H1' and 'V1' in detectors: 
            bad_inx = np.where(((df_temp_snrs['snr_h1']>SNRth_single) & (df_temp_snrs['snr_v1']>SNRth_single) & (df_temp_snrs['observed']>SNRth)) == False)[0]
            df_temp_snrs.loc[bad_inx] = np.nan
            GW_data['H1'] = np.array(df_temp_snrs.snr_h1)
            GW_data['V1'] = np.array(df_temp_snrs.snr_v1)

        if 'L1' and 'V1' in detectors: 
            bad_inx = np.where(((df_temp_snrs['snr_l1']>SNRth_single) & (df_temp_snrs['snr_v1']>SNRth_single) & (df_temp_snrs['observed']>SNRth)) == False)[0]
            df_temp_snrs.loc[bad_inx] = np.nan
            GW_data['L1'] = np.array(df_temp_snrs.snr_l1)
            GW_data['V1'] = np.array(df_temp_snrs.snr_v1)

        if 'H1' and 'L1' and 'V1' in detectors:  
            bad_inx = np.where(((df_temp_snrs['snr_h1']>SNRth_single) & (df_temp_snrs['snr_l1']>SNRth_single) & (df_temp_snrs['snr_v1']>SNRth_single) & (df_temp_snrs['observed']>SNRth)) == False)[0]
            df_temp_snrs.loc[bad_inx] = np.nan
            GW_data['H1'] = np.array(df_temp_snrs.snr_h1)
            GW_data['L1'] = np.array(df_temp_snrs.snr_l1)
            GW_data['V1'] = np.array(df_temp_snrs.snr_v1)
        
        GW_data['snr'] = np.array(df_temp_snrs.observed)
        
    else: 
        GW_data['snr'] = snrs_obs
    
    GW_data['snr'] = snrs_obs

    #get indicies of detected events     
    
    inx_out = np.where((GW_data.snr >= SNRth))[0]   
    GW_data['H0'] = repeated_H0 #add H0 to the GW_data 
    GW_data['gamma'] = repeated_gamma #add gamma to the GW_data 
    GW_data['kappa'] = repeated_kappa #add H0 to the GW_data 
    GW_data['zp'] = repeated_zp #add H0 to the GW_data 
    GW_data['alpha'] = repeated_alpha #add H0 to the GW_data 
    GW_data['beta'] = repeated_beta #add gamma to the GW_data 
    GW_data['mmax'] = repeated_mmax#add H0 to the GW_data 
    GW_data['mmin'] = repeated_mmin #add H0 to the GW_data 
    GW_data['mu_g'] = repeated_mu_g #add H0 to the GW_data 
    GW_data['sigma_g'] = repeated_sigma_g #add gamma to the GW_data 
    GW_data['lambda_peak'] = repeated_lambda#add H0 to the GW_data 
    GW_data['delta_m'] = repeated_delta_m #add H0 to the GW_data
    GW_data['inx'] = inx_gal
    
    if in_out is True:
        GW_data['app_mag'] = app_samples #if using galaxy catalog, get app_mag data 
    else:
        GW_data['app_mag'] = np.ones(len(repeated_H0)) #if not, fill data with ones 
    
    if type_of_data =='testing':
        GW_data['R'] = repeated_Rnums #for tewsting data, get random numbers data 

    if Nselect > 1: #if Nselect was > 1, then check which events to keep from each batch of Nselect
        inds_to_keep = [] #start empty list 
        for k in range(n): #loop over n H0s 
            try: #try to get indicies of where the events where detected every Nselect batches, and just keep one [0]
                inds_to_keep.append(inx_out[(k*int(select) < inx_out) & (inx_out < (k+1)*int(select))][0])
            except IndexError:
                pass #if not, just carry on 
        if len(inds_to_keep) == 0: #if no event was selected, go back to start of loop 
            counter += 1 #add one to counter 
            continue
    else: 
        inds_to_keep = inx_out #else if Nselect was 1, jsut check that one value 
        
    
    out_data = GW_data.loc[np.array(inds_to_keep)] #data to be stored 
    list_data.append(out_data) #append data to lsit 000,802,304
    counter += 1 #add one to counter 
    
    ###Check indicies and thow away observed repeated ones
    if type_of_data == 'training':
        temp_missed_H0 = np.setxor1d(out_data['H0'].to_numpy(),repeated_H0) #check which repeated H0s are not in the stored data and 
        new_missed_H0 = missed_H0[np.where(np.in1d(missed_H0, temp_missed_H0) == True)[0]] #get the missed H0s that have not been detected 
        inx_new_missed = np.where(np.in1d(missed_H0,new_missed_H0) == True) #get the indicies of the missed H0s 
        
    if type_of_data =='testing': #same but doen for random number being used 
        temp_missed_R = np.setxor1d(out_data['R'].to_numpy(),repeated_Rnums)
        new_missed_R = missed_R[np.where(np.in1d(missed_R, temp_missed_R) == True)[0]]
        inx_new_missed = np.where(np.in1d(missed_R,new_missed_R) == True) 
        new_missed_R = missed_R[inx_new_missed] #new missed H0
        inx_missed_R = np.argsort(new_missed_R)
        missed_R = new_missed_R[inx_missed_R]
        

    new_missed_H0 = missed_H0[inx_new_missed] #new missed H0s
    new_missed_gamma = missed_gamma[inx_new_missed] ; new_missed_kappa = missed_kappa[inx_new_missed]; new_missed_zp = missed_zp[inx_new_missed]
    new_missed_alpha = missed_alpha[inx_new_missed] ; new_missed_beta = missed_beta[inx_new_missed]; new_missed_mmax = missed_mmax[inx_new_missed]
    new_missed_mmin = missed_mmin[inx_new_missed] ; new_missed_mu_g = missed_mu_g[inx_new_missed]; new_missed_sigma_g = missed_sigma_g[inx_new_missed]
    new_missed_lambda = missed_lambda[inx_new_missed] ; new_missed_delta_m = missed_delta_m[inx_new_missed]


    inx_missed_H0 = np.argsort(new_missed_H0) #sort the indicies of missed H0s 

    missed_H0 = new_missed_H0[inx_missed_H0] #missed H0s sorted 
    
    missed_gamma = new_missed_gamma[inx_missed_H0] ; missed_kappa = new_missed_kappa[inx_missed_H0]; missed_zp = new_missed_zp[inx_missed_H0]
    missed_alpha = new_missed_alpha[inx_missed_H0] ; missed_beta = new_missed_beta[inx_missed_H0]; missed_mmax = new_missed_mmax[inx_missed_H0]
    missed_mmin = new_missed_mmin[inx_missed_H0] ; missed_mu_g = new_missed_mu_g[inx_missed_H0]; missed_sigma_g = new_missed_sigma_g[inx_missed_H0]
    missed_lambda= new_missed_lambda[inx_missed_H0] ; missed_delta_m = new_missed_delta_m[inx_missed_H0];
    missed_cdfs_zmax = np.array([missed_cdfs_zmax[:,index] for index in inx_missed_H0]).T #get cdfs from each H0s that has been missed 
    missed_cdfs_m1 = xp.array([missed_cdfs_m1[index,:] for index in inx_missed_H0])

    sys.stdout.write('\rEvents we missed: {} | Nselect = {} | counter = {}'.format(len(missed_H0), nxN, counter)) #print status of data generation 
    N_missed = len(missed_H0) #append relevant information 
    counter_list.append(counter)
    Nmissed_list.append(N_missed)
    end = time.time()
    timer_list.append(abs(end - start))

    if len(missed_H0) == 0 : #double check why this is here 
        break

#         temp = []
#         for x in R_nums:
#             temp.append(np.where(missed_R == x)[0])

#         indices_R = np.concatenate(temp,axis=0) 
#         missed_H0 = missed_H0[indices_R]


    sys.stdout.write('\rEvents we missed: {} | Nselect = {} '.format(len(missed_H0), nxN))
    N_missed = len(missed_H0)
    if N_missed == 0 : 
        repeated_H0 = np.repeat(missed_H0, select)
        break


print('\nFINISHED Sampling events')  

#save data 
GW_data = pd.concat(list_data)
output_df = GW_data[['snr', 'H0','gamma','kappa','zp', 'alpha', 'beta', 'mmax', 'mmin', 'mu_g', 'sigma_g', 'lambda_peak', 'delta_m',
                     'luminosity_distance', 'mass_1', 'mass_2', 'ra', 'dec',
                     'a_1', 'a_2', 'tilt_1', 'tilt_2', 'theta_jn',
                     'phi_jl', 'phi_12', 'psi','geocent_time', 'app_mag','inx']]

output_df.to_csv(path_data+'run_{}_det_{}_name_{}_catalog_{}_band_{}_batch_{}_N_{}_SNR_{}_Nelect_{}__Full_para_v1.csv'.format(run,detectors, Name, in_out,band, int(batch), int(N), int(SNRth), int(Nselect)))

if save_timer == 1:
    timer_data = [counter_list,Nmissed_list,timer_list]
    np.savetxt(path_data+'TIMER_run_{}_det_{}_name_{}_catalog_{}_band_{}_batch_{}_N_{}_SNR_{}_Nelect_{}__Full_para_v1.txt'.format(run,detectors, Name, in_out,band, int(batch), int(N), int(SNRth), int(Nselect)), timer_data, delimiter= ',')
    
  
    


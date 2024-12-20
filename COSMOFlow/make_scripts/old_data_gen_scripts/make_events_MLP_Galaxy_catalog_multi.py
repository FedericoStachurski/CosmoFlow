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

from cosmology_functions import priors, cosmology, utilities
from cosmology_functions.zmax import RedshiftGW_fast_zmax
from cosmology_functions.schechter_functions import Schechter_function

from gw_functions import gw_priors_v2
from gw_functions import gw_SNR
from gw_functions.mass_priors import MassPrior_sample
from gw_functions.mass_prior_multi_para_TESTING import MassPrior

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
ap.add_argument("-H0max", "--H0max", required=False,
   help="H0 top boundary", default = 140)
ap.add_argument("-H0min", "--H0min", required=False,
   help="H0 bottom boundary", default = 20)
ap.add_argument("-SNRth", "--SNRth", required=False,
   help="SNR threshold", default = 11)
ap.add_argument("-SNRth_single", "--SNRth_single", required=False,
   help="SNR threshold", default = 0)
ap.add_argument("-band", "--magnitude_band", required=False,
   help="Magnitude band", default = 'K')
ap.add_argument("-run", "--run", required=False,
   help="Detector run [O1, O2]", default = 'O1')
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
ap.add_argument("-H0", "--H0", required=False,
   help="Hubble constant value for testing", default = 70)
ap.add_argument("-fast_zmax", "--fast_zmax", required=False,
   help="Makes the code faster by reducing zmax for specific H0", default = 1)
ap.add_argument("-save_timer", "--save_timer", required=False,
   help="If 1 save agnostic data of data generation, else not", default = 0)
ap.add_argument("-approximator", "--approximator", required=True,
   help="waveform approxiamtor ", default = 'IMRPhenomXPHM')
ap.add_argument("-name_pop", "--name_pop", required=False,
   help="type of population", default = 'BBH')
ap.add_argument("-NSIDE", "--NSIDE", required=True,
   help="NSIDE resolution", default = 32)
ap.add_argument("-targeted", "--targeted", required=True,
   help="Do you want to do targeted event?", default = 'False')

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
H0_testing = float(args['H0'])
save_timer = int(args['save_timer'])
approximator = str(args['approximator'])
population = str(args['name_pop'])
NSIDE = int(args['NSIDE'])
targeted_event = str(args['targeted'])


print()
print('Name model = {}'.format(Name))
print('in_out = {}'.format(in_out))
print('batch = {}'.format(batch))
print('type_of_data = {}'.format(type_of_data))
print('mass_distribution = {}'.format(mass_distribution))
print('zmax = {}'.format(zmax))
print('zmin = {}'.format(zmin))
print('SNRth_combined_network = {}'.format(SNRth))
print('SNRth_single_detector = {}'.format(SNRth_single))
print('mag_band = {}'.format(mag_band))
print('n_detectors = {}'.format(n_det))
print('detectors = {}'.format(detectors))
print('approximator = {}'.format(approximator))
print('run = {}'.format(run))
print('population = {}'.format(population))
print('NSIDE = {}'.format(NSIDE))

if type_of_data == 'training':
    print('H0 = [{},{}]'.format(Hmin, Hmax))
else:    
    print('H0 = {}'.format(H0_testing))

print('N = {}'.format(N))
print('Nselect = {}'.format(Nselect))
print('threads = {}'.format(threads))
print('device = {}'.format(device))
print('fast_zmax = {}'.format(fast_zmax))
print('save_timer= {}'.format(save_timer))
print('seed = {}'.format(seed))
print()


if targeted_event != 'False':
    print('Aiming at event: {}'.format(targeted_event))
    with open('pixel_event/'+str(targeted_event)+'.pickle', 'rb') as file:
        # Load the dictionary from the file
        event_pixels = pickle.load(file)
        pixels_event = event_pixels['pixels']
        NSIDE_event = event_pixels['NSIDE']
        print('NSIDE_event = {}'.format(NSIDE_event))

if run == 'O1':
    model = load_model('models/MLP_models/SNR_MLP_TOTAL_v2_{}_{}_H1_L1/model.pth'.format(approximator, run), device = device) #load MLP model 
    print('SNR approxiamtor = SNR_approximator_{}_{}_H1_L1'.format(approximator, run))
else: 
    if population == 'NSBH':
        model = load_model('models/MLP_models/NSBH_v5_{}_{}_H1_L1_V1/model.pth'.format(approximator, run), device = device) #load MLP 
        print('SNR approxiamtor = SNR_approximator_{}_{}_H1_L1_V1_NSBH'.format(approximator, run))
    else: 
        model = load_model('models/MLP_models/SNR_MLP_TOTAL_v2_{}_{}_H1_L1_V1/model.pth'.format(approximator, run), device = device) #load MLP model 
        print('SNR approxiamtor = SNR_approximator_{}_{}_H1_L1_V1'.format(approximator, run))



indicies_detectors = [] #Check which detectors to use
if 'H1' in detectors: 
    indicies_detectors.append(0)
if 'L1' in detectors:
    indicies_detectors.append(1)
if 'V1' in detectors:
    indicies_detectors.append(2)


in_out = utilities.str2bool(in_out) #cehck if with catalogue or no catalogue
band = mag_band #magnitude band to use 
np.random.seed(seed) # set random seed 

#Load  pixelated galaxy catalog
# NSIDE = 32  #Define NSIDE for healpix map
Npix = hp.nside2npix(NSIDE)

if population == 'BBH':
    name_popualtion_model = 'BBH-powerlaw-gaussian'
elif population == 'NSBH':
    name_popualtion_model = 'NSBH-powerlaw-gaussian'
    
#define popualtion parameters of GWs
# population_parameters = {'beta': 0.81, 'alpha': 3.78,
#                          'mmin': 4.98 ,'mmax': 112.5,
#                          'mu_g': 32.27, 'sigma_g': 3.88,
#                          'lambda_peak': 0.03,
#                          'delta_m': 4.8, 
#                          'name': name_popualtion_model,
#                          'population':population,
#                          'gamma': 4.59, 'k': 2.86, 'zp': 2.47,
#                          'lam': 0,
#                          'Om':0.305, 'zmax':zmax}#put this in sepearte file #Specify 

tolerance = 0.7e-2

beta = xp.array(np.random.uniform(-4,12, N))
alpha = xp.array(np.random.uniform(1.5,12, N))
mmax = xp.array(np.random.uniform(50,200, N))
mmin = xp.array(np.random.uniform(2,10, N))
delta_m = xp.array(np.random.uniform(0,10, N))
lambda_peak = xp.array(np.random.uniform(0,1, N))
mu_g = xp.array(np.random.uniform(20,50, N))
sigma_g = xp.array(np.random.uniform(0.4,10, N))

gamma = xp.array(np.random.uniform(0,12, N))
zp = xp.array(np.random.uniform(0,6, N))
k = xp.array(np.random.uniform(0,4, N))


population_parameters = {'beta': beta, 'alpha': alpha,
                        'mmin': mmin ,'mmax': mmax,
                        'mu_g': mu_g, 'sigma_g': sigma_g,
                        'lambda_peak': lambda_peak,
                        'delta_m': delta_m, 'name': 'BBH-powerlaw-gaussian',
                        'gamma': gamma, 'k': k, 'zp': zp,
                        'lam': 0,
                        'Om':0.305, 'zmax':zmax}


#initiate zmax calss for zmax = f(H0, SNRth) #Hcekc if option is used

z_class = RedshiftGW_fast_z_para(population_parameters, zmin = zmin , zmax = zmax, run = run, SNRth = SNRth, population = population)#initiate zmax calss for zmax 
cdfs_zmax = z_class.make_cdfs()
print('Magic SNRxDL = {}'.format(zmax_class.magic_snr_dl))
mass_class = MassPrior(population_parameters, mgrid = 250)#initiate mass prior class, p(m1,m2)


if in_out is True: #check if using a catalog
    sch_fun = Schechter_function(band) #initiate luminosity functions class 
    
    def load_cat_by_pix(pix): #load pixelated catalog 
        loaded_pix = pd.read_csv('/data/wiay/federico/PhD/cosmoflow/COSMOFlow/pixelated_catalogs/GLADE+_pix_NSIDE_{}/pixel_{}'.format(NSIDE,pix)) #Include NSIDE in the name of folders 
        return loaded_pix
    
    def load_pixel(pix): #load pixel from catalog
        loaded_pix = catalog_pixelated[pix]
        return loaded_pix, len(loaded_pix)

    with multiprocessing.Pool(threads) as p: #begin multiprocesseing for loading the catalog
        catalog_pixelated = list(tqdm(p.imap(load_cat_by_pix,np.arange(Npix)), total = Npix, desc = 'GLADE+ catalog, NSIDE = {}'.format(NSIDE)))

    #load mth map for specific filter band 
    map_mth = np.loadtxt('/data/wiay/federico/PhD/cosmoflow/COSMOFlow/magnitude_threshold_maps/NSIDE_{}_mth_map_GLADE_{}.txt'.format(NSIDE,band))
    inx_0 = np.where(map_mth == 0.0 )[0] #if mag threshold is zero, set it to -inf 
    map_mth[inx_0] = -np.inf
        

def select_gal_from_pix(pixels_H0s): 
    # "Selects galaxies from pixel using pixel index and associated H0 to pixel"
    # "Input: tuple(pixel_inx,H0); 
    # "Returns: dataframe of pixel id (z, ra, dec...)" 
    
    pixel, H0 = pixels_H0s
    loaded_pixel, Ngalpix = load_pixel(int(pixel))
    loaded_pixel = loaded_pixel[['z','RA','dec', 'sigmaz', 'm'+band]] #load pixel 
    loaded_pixel = loaded_pixel.dropna() # drop any Nan values 
    
    temporary_zmax = zmax_class.zmax_H0(H0, SNRth) #for compelteness use the zmax at the given H0 
    
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
        weights_gal = luminosities * zmax_class.time_z(z_gal_selected)

        if np.sum(weights_gal) == 0.0:
            print()
            print('ISSUE: w  ; L ; M ; Dl ; pixel')
            print(weights_gal, luminosities, absolute_mag, dl_galaxies, pixel)
            return 0
        
        weights_gal /= np.sum(weights_gal) # check weights sum to 1
        gal_id = np.random.choice(np.arange(Ngalpix), size = 1, p = weights_gal) #random choice of galaxy in the pixel 
        return loaded_pixel.iloc[gal_id,:]
    
    else: #if no galaxies in pixel, return None 
        return 0 



if type_of_data == 'training':

    #if making training data, define paths where to store the data, and sample H0 
    if in_out is True: 
        path_data = parentdir + r"/data_cosmoflow/galaxy_catalog/training_data_from_MLP/"
    else:
        path_data = parentdir + r"/data_cosmoflow/empty_catalog/training_data_from_MLP/"
        
    H0_samples = np.random.uniform(Hmin,Hmax,N)
    H0_samples = np.sort(H0_samples)
    
    #compute zmax from H0 
    zmax_samples = zmax_class.zmax_H0(H0_samples, SNRth)
    zmax_samples[zmax_samples > zmax] = zmax #if zmax sample is greater than zmax passed, set it equal
    #NO NEED FOR ZMAX INPUT!!!!
    cdfs_zmax = z_class.make_cdfs()
    ### Make p(m1) cdfs
    cdfs_m1 = mass_class.make_cdfs(log_space = False )
    


    
if type_of_data == 'testing': 
    #if making training data, define paths where to store the data, set H0 to specified value 
    if in_out is True: 
        path_data = parentdir + r"/data_cosmoflow/galaxy_catalog/testing_data_from_MLP/"
    else:
        path_data = parentdir + r"/data_cosmoflow/empty_catalog/testing_data_from_MLP/"

    H0_samples = H0_testing*np.ones(N)
    R_nums = np.random.uniform(0,1, size = N) #this is a random number which is used to keep track of which H0 is being used at any given moment 


    
#rename variables (should change this) #Add more comments explainign everything 
H0 = H0_samples 
N_missed = N
list_data = []
## Missed cosmology parameters
missed_H0 = H0 
missed_gamma = gamma 
missed_k = k
missed_zp = zp

## Missed population parameters
missed_beta = beta
missed_alpha = alpha
missed_mmax = mmax
missed_mmin = mmin
missed_delta_m = delta_m
missed_lambda_peak = lambda_peak
missed_mu_g = mu_g
missed_sigma_g = sigma_g


missed_cdfs_zmax = cdfs_zmax
missed_cdfs_m1 = cdfs_zmax

if type_of_data =='testing':
    missed_R = R_nums

counter = 0 #variables to stare for save counter 
counter_list = []
Nmissed_list = []
timer_list = [] #

#begin loop for generating data 
while True:  
    if type_of_data == 'testing':
        repeated_Rnums = np.repeat(missed_R, select) 
        
    start = time.time() # start timer to check efficency in save counter 
    n = len(missed_H0) #check how many H0s are there to be detected
    select = int(Nselect*(N/N_missed)) #use this is the selct value, which increases the more H0s are detected
    nxN = int(n*select) #defin the number of samples we are going to sample nxN (n = H0s, N = sampels per H0s)

    ### Repeated cosmology parameters 
    repeated_H0 = np.repeat(missed_H0, select) #repeat H0s for Nselect samples 
    repeated_gamma = np.repeat(missed_gamma, select)
    repeated_k = np.repeat(missed_zp, select)
    repeated_zp = np.repeat(missed_k, select)

    ### Repeated population parameters
    repeated_beta = np.repeat(missed_beta, select) #repeat betas for Nselect samples 
    repeated_alpha = np.repeat(missed_alpha, select) #repeat alphas for Nselect samples 
    repeated_mmax = np.repeat(missed_mmax, select) #repeat mmaxs for Nselect samples 
    repeated_mmin = np.repeat(missed_mmin, select) #repeat mmins for Nselect samples 
    repeated_delta_m = np.repeat(missed_delta_m, select) #repeat deltams for Nselect samples 
    repeated_lambda_peak = np.repeat(missed_lambda_peak, select) #repeat lambda_peaks for Nselect samples 
    repeated_mu_g = np.repeat(missed_mu_g, select) #repeat mugs for Nselect samples 
    repeated_sigma_g = np.repeat(missed_sigma_g, select) #repeat sigs for Nselect samples 

    inx_gal = np.zeros(nxN) #define galxy indecies 

    if targeted_event != 'False':
        RA, dec = cosmology.target_ra_dec(nxN, pixels_event, NSIDE_event)
    else:     
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
            pixel_H0 = np.array([pix_list, H0_in_list]).T #make an array of lists with pixel index and H0 to be used in the select galaxy function

            with multiprocessing.Pool(threads) as p: #multiprocess for galaxy pixel loading 
                selected_cat_pixels = list(p.imap(select_gal_from_pix,pixel_H0))

            if NSIDE != 32:
                # Checking if any entry is a non-zero DataFrame and keeping only those, while also capturing their indices
                valid_indices = [i for i, df in enumerate(selected_cat_pixels) if isinstance(df, pd.DataFrame) and not df.empty]

                # Now valid_indices contains the indices of selected_cat_pixels where the entry is an actual non-empty DataFrame
                # If you also need to keep only the actual non-empty DataFrames:
                selected_cat_pixels = [df for df in selected_cat_pixels if isinstance(df, pd.DataFrame) and not df.empty]
                H0_in_list = H0_in_list[valid_indices]
                inx_in_gal = inx_in_gal[valid_indices]
            
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

   
    #sample priors on theta_jn, psi, geo_time
    #_, _, _, _, _, _, _, _, _, theta_jn, _, _, psi, _ , geo_time = gw_priors_v2.draw_prior(int(nxN))
    _, _, _, a1, a2, tilt1, tilt2, _, _, theta_jn, phi_jl, phi_12, psi, _, geo_time = gw_priors_v2.draw_prior(int(nxN))
    
    #Sample primary and secondary masses
    m1 = mass_class.draw_m(select, missed_cdfs_m1, m_array_long = None, log_space = False)
    valid_samples = xp.array(samples_m1 >= repeated_mmin)
    while not xp.all(valid_samples):
        invalid_indices = xp.where(~valid_samples)[0]
        new_samples = mass_class.draw_m(1, missed_cdfs_m1[invalid_indices // select, :])
        samples_m1[invalid_indices.get()] = new_samples
        valid_samples = xp.array((samples_m1 >=repeated_mmin))
    
    m_vect_m2 = xp.linspace(mmin.get()[:, None], m1[:, None], mass_class.mgrid, axis=1)
    m_vect_m2 = xp.reshape(m_vect_m2, (len(m1), mass_class.mgrid))
    pdfs_m2 = mass_class.powerlaw_smooth_m2_vect(m_vect_m2,xp.array(m1), beta=repeated_beta,
                                             mmin=repeated_mmin,
                                             delta_m=repeated_delta_m)
    cdf_m2 = xp.cumsum(pdfs_m2, axis=1)
    cdf_maximum_m2 = xp.amax(cdf_m2, axis=1)[:, None]
    cdf_m2 /= cdf_maximum_m2
    m2 = mass_class.draw_m(1, cdf_m2, m_array_long=None, log_space=False, m2_arr = np.hstack(m_vect_m2))

    
    m1z = m1*(1+z) #turn source masses into detector frame masses 
    m2z = m2*(1+z)
    
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
        
    #get indicies of detected events     
    inx_out = np.where((GW_data.snr >= SNRth))[0]   
    GW_data['H0'] = repeated_H0 #add H0 to the GW_data 
    GW_data['z'] = z #add H0 to the GW_data 
    
    if in_out is True:
        GW_data['app_mag'] = app_samples #if using galaxy catalog, get app_mag data 
    else:
        GW_data['app_mag'] = np.ones(len(repeated_H0)) #if not, fill data with ones 
    GW_data['inx'] = inx_gal

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
    list_data.append(out_data) #append data to lsit 
    counter += 1 #add one to counter 

    if type_of_data =='training':
        temp_missed_H0 = np.setxor1d(out_data['H0'].to_numpy(),repeated_H0) #check which repeated H0s are not in the stored data and 
        new_missed_H0 = missed_H0[np.where(np.in1d(missed_H0, temp_missed_H0) == True)[0]] #get the missed H0s that have not been detected 
        inx_new_missed = np.where(np.in1d(missed_H0,new_missed_H0) == True) #get the indicies of the missed H0s 
        
        new_missed_H0 = missed_H0[inx_new_missed] #new missed H0s
        inx_missed_H0 = np.argsort(new_missed_H0) #sort the indicies of missed H0s 
        missed_H0 = new_missed_H0[inx_missed_H0] #missed H0s sorted 
        missed_cdfs_zmax = [missed_cdfs_zmax[index] for index in inx_missed_H0] #get cdfs from each H0s that has been missed to be used in next iteration

        sys.stdout.write('\rH0 we missed: {} | Nselect = {} | counter = {}'.format(len(missed_H0), nxN, counter-1)) #print status of data generation 
        N_missed = len(missed_H0) #append relevant information 
        counter_list.append(counter)
        Nmissed_list.append(N_missed)
        end = time.time()
        timer_list.append(abs(end - start))
        
        if len(missed_H0) == 0 : #double check why this is here 
            repeated_H0 = np.repeat(missed_H0, select)
            break

    elif type_of_data =='testing': #same but doen for random number being used 
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

#save data 
GW_data = pd.concat(list_data)
output_df = GW_data[['snr', 'H0', 'luminosity_distance', 'mass_1', 'mass_2', 'ra', 'dec',
                     'a_1', 'a_2', 'tilt_1', 'tilt_2', 'theta_jn',
                     'phi_jl', 'phi_12', 'psi','geocent_time', 'app_mag', 'inx', 'z']]

output_df.to_csv(path_data+'run_{}_det_{}_name_{}_catalog_{}_band_{}_batch_{}_N_{}_SNR_{}_Nelect_{}__Full_para_v1.csv'.format(run,detectors, Name, in_out,band, int(batch), int(N), int(SNRth), int(Nselect)))

if save_timer == 1:
    timer_data = [counter_list,Nmissed_list,timer_list]
    np.savetxt(path_data+'TIMER_run_{}_det_{}_name_{}_catalog_{}_band_{}_batch_{}_N_{}_SNR_{}_Nelect_{}__Full_para_v1.txt'.format(run,detectors, Name, in_out,band, int(batch), int(N), int(SNRth), int(Nselect)), timer_data, delimiter= ',')
    
  
    


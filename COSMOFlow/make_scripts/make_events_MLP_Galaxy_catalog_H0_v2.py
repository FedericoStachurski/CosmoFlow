#%% import libraries
import os
import sys
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

# Get the current script directory
current_dir = os.path.dirname(os.path.realpath(__file__))

# Get the parent directory (where cosmology_functions is located)
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)

# Custom modules
from cosmology_functions import priors, cosmology, utilities
from cosmology_functions.zmax import RedshiftGW_fast_zmax
from cosmology_functions.schechter_functions import Schechter_function
from gw_functions import gw_priors_v2
from gw_functions import gw_SNR
from gw_functions.mass_priors import MassPrior_sample

import warnings
warnings.filterwarnings("ignore")

import argparse
import random
import h5py 
import healpy as hp
import multiprocessing
from scipy.stats import truncnorm, chi2, ncx2
from MultiLayerPerceptron.validate import run_on_dataset
from MultiLayerPerceptron.nn.model_creation import load_mlp
from poplar.nn.networks import LinearModel, load_model
from poplar.nn.rescaling import ZScoreRescaler
import torch
import cupy as cp

xp = cp  # Placeholder for cupy (GPU)

# ================== Argument Parsing ==================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate training data of GW observables with H0 values.")

    parser.add_argument("-Name", "--Name_file", required=True, help="Name of the data")
    parser.add_argument("-in_out", "--in_out", required=True, help="With catalog (1) or without (0)", default=1)
    parser.add_argument("-batch", "--batch_number", required=True, help="Batch number of the data", default=1)
    parser.add_argument("-type", "--type_data", required=True, help="Type of data: training or testing", default='training')
    parser.add_argument("-mass_distribution", "--mass_distribution", help="Mass distribution options: [uniform, PowerLaw+Peak, PowerLaw]", default='PowerLaw+Peak')
    parser.add_argument("-zmax", "--zmax", help="Maximum redshift", default=1.5)
    parser.add_argument("-zmin", "--zmin", help="Minimum redshift", default=0.0001)
    parser.add_argument("-H0max", "--H0max", help="Upper boundary for H0", default=140)
    parser.add_argument("-H0min", "--H0min", help="Lower boundary for H0", default=20)
    parser.add_argument("-SNRth", "--SNRth", help="SNR threshold", default=11)
    parser.add_argument("-SNRth_single", "--SNRth_single", help="SNR threshold for individual detectors", default=0)
    parser.add_argument("-band", "--magnitude_band", help="Magnitude band", default='K')
    parser.add_argument("-run", "--run", help="Detector run [O1, O2]", default='O1')
    parser.add_argument("-detectors", "--detectors", nargs='+', required=True, help="Detectors: [H1, L1, V1]", default=['H1'])
    parser.add_argument("-N", "--N", required=True, help="Samples per batch", default=100_000)
    parser.add_argument("-Nselect", "--Nselect", help="Nselect per iteration", default=5)
    parser.add_argument("-threads", "--threads", help="Number of threads", default=10)
    parser.add_argument("-device", "--device", help="Device: [cpu, cuda]", default='cpu')
    parser.add_argument("-seed", "--seed", help="Random seed", default=1234)
    parser.add_argument("-H0", "--H0", help="Hubble constant value for testing", default=70)
    parser.add_argument("-fast_zmax", "--fast_zmax", help="Fast zmax option (1 for True)", default=1)
    parser.add_argument("-save_timer", "--save_timer", help="Save timing data (1 to save)", default=0)
    parser.add_argument("-approximator", "--approximator", required=True, help="Waveform approximator", default='IMRPhenomXPHM')
    parser.add_argument("-name_pop", "--name_pop", help="Population type", default='BBH')
    parser.add_argument("-NSIDE", "--NSIDE", required=True, help="NSIDE resolution", default=32)
    parser.add_argument("-targeted", "--targeted", help="Targeted event (True/False)", default='False')

    return vars(parser.parse_args())

# Call the argument parser
args = parse_arguments()

# Assign variables from the parsed arguments
Name = args['Name_file']
in_out = args['in_out']
batch = int(args['batch_number'])
type_of_data = args['type_data']
mass_distribution = args['mass_distribution']
zmax = float(args['zmax'])
zmin = float(args['zmin'])
Hmax = float(args['H0max'])
Hmin = float(args['H0min'])
SNRth = float(args['SNRth'])
SNRth_single = float(args['SNRth_single'])
mag_band = args['magnitude_band']
N = int(args['N'])
Nselect = int(args['Nselect'])
threads = int(args['threads'])
run = args['run']
detectors = args['detectors']
device = args['device']
fast_zmax = int(args['fast_zmax'])
seed = int(args['seed'])
H0_testing = float(args['H0'])
save_timer = int(args['save_timer'])
approximator = args['approximator']
population = args['name_pop']
NSIDE = int(args['NSIDE'])
targeted_event = args['targeted']

# Count the number of detectors
n_det = len(detectors)



# ========== Setup and Validation ==========

# Set the random seed
np.random.seed(seed)

# Validate or print parsed parameters
# Print and validate all parsed parameters
print("\n===== PARAMETERS =====")
print(f"Name of the model: {Name}")
print(f"Type of data: {type_of_data} (Options: training or testing)")
print(f"With catalog: {in_out} (1: Yes, 0: No)")
print(f"Batch number: {batch}")
print(f"Mass distribution: {mass_distribution} (Options: uniform, PowerLaw+Peak, PowerLaw)")
print(f"Redshift range: zmin = {zmin}, zmax = {zmax}")
print(f"Hubble constant range for training: H0min = {Hmin}, H0max = {Hmax}")
print(f"Testing Hubble constant value: {H0_testing} (Used only for testing)")
print(f"SNR threshold: Combined network = {SNRth}, Single detector = {SNRth_single}")
print(f"Magnitude band: {mag_band}")
print(f"Detector run: {run} (Options: O1, O2)")
print(f"Detectors selected: {detectors} (Total: {len(detectors)})")
print(f"Number of samples (per batch): {N}")
print(f"Samples per iteration (Nselect): {Nselect}")
print(f"Number of threads: {threads}")
print(f"Device for computation: {device} (Options: cpu, cuda)")
print(f"Fast zmax option: {fast_zmax} (1: Enabled, 0: Disabled)")
print(f"Random seed: {seed}")
print(f"Save timing data: {save_timer} (1: Enabled, 0: Disabled)")
print(f"Waveform approximator: {approximator}")
print(f"Population type: {population} (Options: BBH, NSBH)")
print(f"Healpix NSIDE resolution: {NSIDE}")
print(f"Targeted event: {targeted_event} (False if not targeted)")
print("======================\n")

# Additional validation for training/testing specific parameters
if type_of_data == 'training':
    print(f"Training mode: Hubble constant range set to [{Hmin}, {Hmax}]")
else:
    print(f"Testing mode: Hubble constant set to {H0_testing}")

# Handle Targeted Event
if targeted_event != 'False':
    print(f'Aiming at event: {targeted_event}')
    targeted_event_path = os.path.join('pixel_event', f'{targeted_event}.pickle')
    
    try:
        with open(targeted_event_path, 'rb') as file:
            # Load the dictionary from the file
            event_pixels = pickle.load(file)
            pixels_event = event_pixels['pixels']
            NSIDE_event = event_pixels['NSIDE']
            print(f'NSIDE_event = {NSIDE_event}')
    except FileNotFoundError:
        print(f"Error: Event file for {targeted_event} not found at {targeted_event_path}.")
        sys.exit(1)  # Exit if the file is not found

# Load the appropriate MLP model
if run == 'O1':
    model_path = f'models/MLP_models/SNR_MLP_TOTAL_v2_{approximator}_{run}_H1_L1/model.pth'
    print(f'Loading SNR approximator model from {model_path}')
else:
    if population == 'NSBH':
        model_path = f'models/MLP_models/NSBH_v5_{approximator}_{run}_H1_L1_V1/model.pth'
        print(f'Loading SNR approximator for NSBH from {model_path}')
    else:
        model_path = f'models/MLP_models/SNR_MLP_TOTAL_v2_{approximator}_{run}_H1_L1_V1/model.pth'
        print(f'Loading SNR approximator for BBH from {model_path}')

# Load the model
try:
    model = load_model(model_path, device=device)  # Load MLP model
except Exception as e:
    print(f"Error loading model: {str(e)}")
    sys.exit(1)  # Exit if model fails to load

# Check which detectors to use
indicies_detectors = []  # Initialize empty list
if 'H1' in detectors: 
    indicies_detectors.append(0)
if 'L1' in detectors:
    indicies_detectors.append(1)
if 'V1' in detectors:
    indicies_detectors.append(2)

print(f"Using detectors: {detectors}, indices: {indicies_detectors}")

# Convert the 'in_out' string argument to boolean. This determines whether to use the galaxy catalog or not.
# The utilities.str2bool function converts strings like "1" to True, and "0" to False.
try:
    in_out = utilities.str2bool(in_out)  # Convert string to boolean
except Exception as e:
    # If the conversion fails (for example, due to an invalid input), we catch the error and stop the program.
    print(f"Error converting 'in_out' to boolean: {e}")
    sys.exit(1)

# Set the magnitude band for the simulation, e.g., 'K' for the K-band of light.
band = mag_band  

# Set a fixed seed for the random number generator to ensure reproducibility of results.
np.random.seed(seed)

# Determine the total number of pixels for the Healpix map based on the NSIDE resolution.
# Healpix is a pixelization scheme commonly used for sky maps.
Npix = hp.nside2npix(NSIDE)
print(f"Healpix NSIDE = {NSIDE}, Total pixels = {Npix}")

# Determine the name of the population model based on the type of population (BBH or NSBH).
# This model name is used to label and differentiate between different simulations.
if population == 'BBH':
    name_population_model = 'BBH-powerlaw-gaussian'
elif population == 'NSBH':
    name_population_model = 'NSBH-powerlaw-gaussian'
else:
    # If the provided population type is not recognized, we print an error message and exit the program.
    print(f"Unknown population type: {population}")
    sys.exit(1)

# Define a dictionary of population parameters for the gravitational wave events.
# These parameters include distributions for masses, redshifts, and cosmological constants.
# The parameters are currently hardcoded but could later be moved to a configuration file for flexibility.
# //////////////////////// POPULATION PARAMETERS /////////////////////////////////////////
population_parameters = {
    'beta': 0.81,                # Power-law index for mass distribution
    'alpha': 3.78,               # Power-law index for redshift evolution
    'mmin': 4.98,                # Minimum mass of compact objects (solar masses)
    'mmax': 112.5,               # Maximum mass of compact objects (solar masses)
    'mu_g': 32.27,               # Mean of Gaussian component in mass distribution (solar masses)
    'sigma_g': 3.88,             # Standard deviation of Gaussian component in mass distribution
    'lambda_peak': 0.03,         # Fraction of Gaussian component in mass distribution
    'delta_m': 4.8,              # Sharpness of the mass cutoff in the distribution
    'name': name_population_model,  # Name of the population model (BBH or NSBH)
    'population': population,    # Type of population ('BBH' or 'NSBH')
    'gamma': 4.59,               # Parameter for rate evolution with redshift
    'k': 2.86,                   # Parameter for luminosity evolution
    'zp': 2.47,                  # Redshift at which the rate evolution changes slope
    'lam': 0,                    # Lambda parameter (related to cosmology)
    'Om': 0.305,                 # Omega_matter, a cosmological parameter (density of matter in the universe)
    'zmax': zmax                 # Maximum redshift for the simulation
}

# Initialize the RedshiftGW_fast_zmax class, which calculates the maximum redshift for gravitational wave events.
# This class will use the population parameters and the specified redshift range to compute various metrics.
zmax_class = RedshiftGW_fast_zmax(
    parameters=population_parameters,  # Pass the population parameters
    run=run,                           # Specify the detector run (O1, O2, etc.)
    zmin=zmin,                         # Minimum redshift considered in the simulation
    zmax=zmax                          # Maximum redshift considered in the simulation
)
# Print the value of the magic SNR x DL, a constant used in calculating signal-to-noise ratios for GW detections.
print(f"Magic SNRxDL = {zmax_class.magic_snr_dl}")

# Initialize the MassPrior_sample class, which samples masses for gravitational wave events
# based on the specified mass distribution (e.g., PowerLaw+Peak, Uniform).
# This class will generate the masses for the simulated events.
try:
    mass_class = MassPrior_sample(population_parameters, mass_distribution)
    print(f"Mass prior class initialized for {mass_distribution} distribution.")
except Exception as e:
    # If the mass prior class fails to initialize (for example, due to an invalid parameter),
    # we catch the error, print a message, and exit the program.
    print(f"Error initializing mass prior sampling class: {e}")
    sys.exit(1)


if in_out:  # Check if we are using a galaxy catalog (if 'in_out' is True)
    # Initialize the Schechter function class, which is used to model the luminosity functions of galaxies
    sch_fun = Schechter_function(band)  # The magnitude band (e.g., 'K') is passed as an argument

    # Define a helper function to load the pixelated galaxy catalog for a specific pixel
    def load_cat_by_pix(pix):
        # Construct the file path for each pixel in the catalog
        catalog_path = '/data/wiay/federico/PhD/cosmoflow/COSMOFlow/pixelated_catalogs/GLADE+_pix_NSIDE_{}/pixel_{}'.format(NSIDE, pix)
        try:
            # Attempt to load the pixel's catalog as a pandas DataFrame
            loaded_pix = pd.read_csv(catalog_path)
        except FileNotFoundError:
            # If the file is not found, print a message and return an empty DataFrame
            print(f"Error: Galaxy catalog for pixel {pix} not found at {catalog_path}.")
            loaded_pix = pd.DataFrame()  # Return an empty DataFrame in case of failure
        return loaded_pix

    # Define a function to retrieve the loaded catalog for a specific pixel
    def load_pixel(pix):
        # Access the preloaded catalog for the given pixel index from the catalog_pixelated list
        loaded_pix = catalog_pixelated[pix]
        # Return the loaded pixel's data and the length of the DataFrame (number of galaxies in the pixel)
        return loaded_pix, len(loaded_pix)

    # Load the entire galaxy catalog in parallel using multiprocessing
    # Use the number of threads defined by the 'threads' argument for parallel processing
    with multiprocessing.Pool(threads) as p:
        # 'tqdm' is used to show a progress bar for the catalog loading process
        catalog_pixelated = list(tqdm(p.imap(load_cat_by_pix, np.arange(Npix)), 
                                      total=Npix, 
                                      desc=f'Loading GLADE+ catalog, NSIDE = {NSIDE}'))

    # Load the magnitude threshold map for the specified magnitude band and NSIDE resolution
    # This map contains the limiting magnitude values for each pixel in the catalog
    mth_map_path = '/data/wiay/federico/PhD/cosmoflow/COSMOFlow/magnitude_threshold_maps/NSIDE_{}_mth_map_GLADE_{}.txt'.format(NSIDE, band)
    
    try:
        # Attempt to load the magnitude threshold map as a NumPy array
        map_mth = np.loadtxt(mth_map_path)
    except FileNotFoundError:
        # If the file is not found, print an error message and exit the program
        print(f"Error: Magnitude threshold map not found at {mth_map_path}.")
        sys.exit(1)

    # Find indices where the magnitude threshold is zero and replace those values with -infinity
    # A magnitude threshold of zero indicates that no observations were made in that pixel
    inx_0 = np.where(map_mth == 0.0)[0]
    map_mth[inx_0] = -np.inf  # Replace zero thresholds with -inf to exclude those pixels from further consideration

    print("Galaxy catalogue and magnitude threshold map loaded successfully.")
else: 
    print("Generating GW events with no galaxy catalogue information")


#Define function for glaaxy seleciton in pixel catalogue
def select_gal_from_pix(pixels_H0s):
    """
    Selects galaxies from a pixel based on its index and the associated Hubble constant (H0).
    
    Args:
        pixels_H0s (tuple): A tuple containing the pixel index and H0 (Hubble constant).
    
    Returns:
        DataFrame row: A single row from the loaded galaxy DataFrame, corresponding to the selected galaxy,
                       or 0 if no galaxy is selected.
    """
    
    # Unpack the tuple to get the pixel index and H0 value
    pixel, H0 = pixels_H0s

    # Load the galaxy catalog for the specified pixel using the load_pixel function
    loaded_pixel, Ngalpix = load_pixel(int(pixel))
    
    # Select only the necessary columns for further processing
    # 'z' is the redshift, 'RA' is the right ascension, 'dec' is the declination, 'sigmaz' is the redshift error,
    # and 'm+band' is the apparent magnitude in the specified band (e.g., K band).
    loaded_pixel = loaded_pixel[['z', 'RA', 'dec', 'sigmaz', 'm'+band]]
    
    # Drop any rows with missing (NaN) values
    loaded_pixel = loaded_pixel.dropna()

    # Calculate the maximum redshift (zmax) based on H0 and SNR threshold (SNRth)
    # This ensures that galaxies beyond this redshift are not considered.
    temporary_zmax = zmax_class.zmax_H0(H0, SNRth)

    # Filter galaxies based on redshift constraints: z must be between zmin and zmax
    loaded_pixel = loaded_pixel[loaded_pixel.z <= temporary_zmax]
    loaded_pixel = loaded_pixel[loaded_pixel.z >= zmin]

    # Convert right ascension (RA) and declination (dec) from degrees to radians
    loaded_pixel['RA'] = np.deg2rad(loaded_pixel['RA'])
    loaded_pixel['dec'] = np.deg2rad(loaded_pixel['dec'])

    # Update the number of galaxies in the pixel after filtering
    Ngalpix = len(loaded_pixel)

    # If the pixel contains galaxies, proceed to select one
    if not loaded_pixel.empty:
        # Extract redshift of galaxies
        z_gal_selected = loaded_pixel['z']
        
        # Create an array with repeated H0 values (one for each galaxy in the pixel)
        repeated_H0_in_pix = np.ones(Ngalpix) * H0
        
        # Compute the luminosity distance (Dl) of galaxies using their redshift and H0
        dl_galaxies = cosmology.fast_z_to_dl_v2(np.array(z_gal_selected).flatten(), 
                                                np.array(repeated_H0_in_pix).flatten())
        
        # Compute the absolute magnitudes of galaxies using their apparent magnitudes and distances
        absolute_mag = cosmology.abs_M(loaded_pixel['m'+band], dl_galaxies)
        
        # Convert the absolute magnitudes into luminosities
        luminosities = cosmology.mag2lum(absolute_mag)
        
        # Calculate galaxy weights based on their luminosity, redshift evolution (madau), and time factors
        weights_gal = luminosities * zmax_class.time_z(z_gal_selected)
        
        # Check if the sum of weights is zero (indicating an issue with the data)
        if np.sum(weights_gal) == 0.0:
            print("\nISSUE: w  ; L ; M ; Dl ; pixel")
            print(weights_gal, luminosities, absolute_mag, dl_galaxies, pixel)
            return 0  # Return 0 to indicate failure in this pixel

        # Normalize the weights so that they sum to 1
        weights_gal /= np.sum(weights_gal)
        
        # Randomly select a galaxy based on the computed weights
        gal_id = np.random.choice(np.arange(Ngalpix), size=1, p=weights_gal)
        
        # Return the selected galaxy as a DataFrame row
        return loaded_pixel.iloc[gal_id, :]
    
    else:
        # If the pixel contains no galaxies, return 0 to indicate failure
        return 0

#Check type of data we are generating
# If generating training data
if type_of_data == 'training':

    # Define the path to store the generated training data
    # If using a galaxy catalog, store the data in the 'galaxy_catalog' folder
    if in_out:
        path_data = os.path.join(parent_dir, "data_cosmoflow", "galaxy_catalog", "training_data_from_MLP")
    # Otherwise, store the data in the 'empty_catalog' folder
    else:
        path_data = os.path.join(parent_dir, "data_cosmoflow", "empty_catalog", "training_data_from_MLP")
    
    # Sample H0 values from a uniform distribution between Hmin and Hmax, for a total of N samples
    H0_samples = np.random.uniform(Hmin, Hmax, N)
    # Sort the sampled H0 values in ascending order
    H0_samples = np.sort(H0_samples)
    
    # Compute the maximum redshift (zmax) for each H0 value based on the SNR threshold (SNRth)
    zmax_samples = zmax_class.zmax_H0(H0_samples, SNRth)
    # If any zmax values exceed the global zmax, set them to the global zmax
    zmax_samples[zmax_samples > zmax] = zmax
    
    # Generate cumulative distribution functions (cdfs) from the zmax samples using multiprocessing
    # 'make_cdfs' is a method of the zmax_class that computes the cdf for each zmax value
    # tqdm provides a progress bar for the multiprocessing operation
    with multiprocessing.Pool(threads) as p:
        cdfs_zmax = list(tqdm(p.imap(zmax_class.make_cdfs, zmax_samples), 
                              total=N, 
                              desc='Making cdfs from p(z)p(s|z)'))

# If generating testing data
elif type_of_data == 'testing':
    
    # Define the path to store the generated testing data
    # Use different directories for galaxy catalog and empty catalog cases
    if in_out:
        path_data = os.path.join(parentdir, "data_cosmoflow", "galaxy_catalog", "testing_data_from_MLP")
    else:
        path_data = os.path.join(parentdir, "data_cosmoflow", "empty_catalog", "testing_data_from_MLP")
    
    # For testing, we use a fixed H0 value (H0_testing) for all N samples
    H0_samples = H0_testing * np.ones(N)
    
    # Generate a random number between 0 and 1 for each H0 sample.
    # These random numbers are used to track which H0 value is being used during testing.
    R_nums = np.random.uniform(0, 1, size=N)


# Hubble constant samples (H0) and their corresponding cumulative distribution functions (CDFs)
missed_H0 = H0_samples  # Initially, all H0 values are considered "missed" (i.e., unprocessed)
missed_cdfs_zmax = cdfs_zmax  # CDFs for the corresponding H0 values

# If in testing mode, also track the random numbers associated with each H0 sample
missed_R = R_nums if type_of_data == 'testing' else None

# Tracking variables
N_missed = N  # Start with all H0 samples as "missed" (unprocessed)
list_data = []  # To store the generated data

# Initialize iteration counters and tracking lists
counter = 0  # Track the number of iterations
counter_list = []  # Store iteration counts
Nmissed_list = []  # Track how many H0 samples remain unprocessed per iteration
timer_list = []  # Store timing information for each iteration



# Start the loop for generating data
while True:
    if type_of_data == 'testing':
        # Repeat the random numbers associated with H0 for each sample
        repeated_Rnums = np.repeat(missed_R, select) 
        
    start = time.time()  # Start timer to track efficiency
    n = len(missed_H0)  # Number of H0 samples that haven't been processed yet
    # The selection value increases as more H0 values are detected
    select = int(Nselect * (N / N_missed))  # Define the select value based on how many H0 values are missed
    nxN = int(n * select)  # Define the number of samples we're going to generate
    repeated_H0 = np.repeat(missed_H0, select)  # Repeat H0s for Nselect samples
    inx_gal = np.zeros(nxN)  # Initialize array for galaxy indices (will store 1 if a galaxy is selected)

    ######### Sample RA and Dec ###############
    if targeted_event != 'False':
        # If a targeted event is specified, use RA/Dec for that event
        RA, dec = cosmology.target_ra_dec(nxN, pixels_event, NSIDE_event)
    else:
        # Otherwise, randomly sample RA and Dec values
        RA, dec = cosmology.draw_RA_Dec(nxN)
    
    # Sample redshifts based on zmax-H0 distributions for each missed H0
    z = zmax_class.draw_z_zmax(select, missed_cdfs_zmax)
    # Convert redshifts and H0 values into luminosity distances
    dl = cosmology.fast_z_to_dl_v2(np.array(z), np.array(repeated_H0))
    
    # Ensure all variables are arrays
    z = np.array(z)
    dl = np.array(dl)

    # If using the galaxy catalog (in_out == True)
    if in_out:

        #### M and m #######
        # Sample absolute magnitudes from the Schechter function for H0 = 100
        M_abs = sch_fun.sample_M_from_cdf_weighted(100, N=nxN)
        # Adjust the magnitudes based on the H0 values
        M_abs = M_abs + 5 * np.log10(repeated_H0 / 100)
        # Compute apparent magnitudes from absolute magnitudes and distances
        app_samples = cosmology.app_mag(M_abs.flatten(), dl.flatten())
        
        # Check if we're using a galaxy catalog. If 'in_out' is True, this indicates we're working with a galaxy catalog.
        # Handle the magnitude threshold (mth) map
        # We map the right ascension (RA) and declination (dec) values to their corresponding magnitude threshold (mth)
        # for each galaxy. This mth_map is used to filter out galaxies that are too faint to detect.
        # 'mth_from_RAdec' converts the RA/Dec values to the corresponding threshold value for each galaxy.
        mth_list = np.array([utilities.mth_from_RAdec(NSIDE, RA, dec, map_mth)]).flatten()
    
        # Find the pixel for each RA/Dec coordinate in the Healpix map using 'pix_from_RAdec'.
        # This maps the galaxy's position on the sky to a specific pixel in the Healpix representation.
        pix_list = np.array([utilities.pix_from_RAdec(NSIDE, RA, dec)]).flatten()
    
        # Select galaxies whose apparent magnitude (app_samples) is brighter than the magnitude threshold (mth_list).
        # This filters out galaxies that are too dim for detection (those with an apparent magnitude greater than the threshold).
        inx_in_gal = np.where(app_samples < mth_list)[0]  # Keep indices where app_samples are less than the threshold
    
        # Check if we found any galaxies that are bright enough to be considered for selection
        if len(inx_in_gal) > 0:
            # Extract the pixel list and the associated Hubble constant (H0) values for galaxies that passed the magnitude threshold check.
            pix_list = np.array(pix_list[inx_in_gal])
            H0_in_list = np.array(repeated_H0[inx_in_gal])
    
            # Create a list of tuples (pixel index, corresponding H0) to pass to the galaxy selection function
            pixel_H0 = np.array([pix_list, H0_in_list]).T
    
            # Use multiprocessing to speed up the process of selecting galaxies from the catalog
            # The function 'select_gal_from_pix' will be applied to each (pixel, H0) tuple
            # This allows parallel processing of the galaxy catalog using the specified number of threads.
            with multiprocessing.Pool(threads) as p:
                selected_cat_pixels = list(p.imap(select_gal_from_pix, pixel_H0))
    
            # If NSIDE is not equal to 32, perform further validation
            # We are checking for entries in 'selected_cat_pixels' that are valid (non-empty DataFrames).
            if NSIDE != 32:
                # Identify valid indices, i.e., where the selected catalog pixels are valid DataFrames (not empty)
                valid_indices = [i for i, df in enumerate(selected_cat_pixels) if isinstance(df, pd.DataFrame) and not df.empty]
                
                # Keep only the non-empty DataFrames from the selected galaxies
                selected_cat_pixels = [df for df in selected_cat_pixels if isinstance(df, pd.DataFrame) and not df.empty]
                
                # Update the H0 and index arrays to match the valid (non-empty) galaxies
                H0_in_list = H0_in_list[valid_indices]
                inx_in_gal = inx_in_gal[valid_indices]
    
            # If we have selected one or more galaxies, proceed to extract relevant information
            if len(selected_cat_pixels) >= 1:
                # Concatenate all the selected galaxies into a single DataFrame
                gal_selected = pd.concat(selected_cat_pixels)
    
                # Extract the RA, Dec, redshift (z), and redshift uncertainty (sigmaz) from the selected galaxies
                RA_gal = np.array(gal_selected.RA)      # Right Ascension
                dec_gal = np.array(gal_selected.dec)    # Declination
                z_true_gal = np.array(gal_selected.z)   # True redshift
                sigmaz_gal = np.array(gal_selected.sigmaz)  # Redshift uncertainty
    
                # Sample observed redshifts using a truncated Gaussian distribution based on redshift uncertainty (sigmaz)
                # We use the limits zmin and zmax to restrict the range of possible redshift values.
                a, b = (zmin - z_true_gal) / sigmaz_gal, (zmax - z_true_gal) / sigmaz_gal
                z_obs_gal = truncnorm.rvs(a, b, loc=z_true_gal, scale=abs(sigmaz_gal), size=len(z_true_gal))
    
                # Extract observed apparent magnitudes for the selected galaxies
                m_obs_gal = np.array(gal_selected['m' + band])
    
                # Calculate the luminosity distance for the galaxies using the observed redshift and the corresponding H0 values
                dl_gal = cosmology.fast_z_to_dl_v2(np.array(z_obs_gal), np.array(H0_in_list))
    
                # Replace the initially sampled redshifts and distances in the original arrays with the values for the selected galaxies
                z[inx_in_gal] = z_obs_gal
                dl[inx_in_gal] = dl_gal
                RA[inx_in_gal] = RA_gal
                dec[inx_in_gal] = dec_gal
                app_samples[inx_in_gal] = m_obs_gal  # Replace apparent magnitudes with those of the selected galaxies
                inx_gal[inx_in_gal] = 1  # Mark these galaxies as having been selected by setting the index to 1


    ############# GW Intrinsic parameters #####################################
    # Sample priors on gravitational wave parameters
    # theta_jn (inclination angle), psi (polarization angle), geo_time (geocentric time) are some of the sampled parameters
    _, _, _, a1, a2, tilt1, tilt2, _, _, theta_jn, phi_jl, phi_12, psi, _, geo_time = gw_priors_v2.draw_prior(int(nxN))
    
    # Sample primary (m1) and secondary (m2) masses for the gravitational wave events
    # The 'PL_PEAK_GWCOSMO' function samples from a mass distribution based on population priors for GW events
    m1, m2 = mass_class.PL_PEAK_GWCOSMO(nxN)  # Sample primary and secondary masses based on GWCOSMO mass prior
    
    # Adjust the source masses for redshift (detector frame masses)
    # The observed masses are scaled by (1 + z) to account for the redshift of the sources.
    m1z = m1 * (1 + z)  # Mass of the primary black hole in detector frame
    m2z = m2 * (1 + z)  # Mass of the secondary black hole in detector frame
    
    # Create a dictionary to store the gravitational wave parameters
    # This dictionary includes the sampled masses, spins, orientation angles, sky location, and luminosity distance
    data_dict = {
        'luminosity_distance': dl,  # Luminosity distance in Mpc
        'mass_1': m1z,  # Detector-frame mass of the primary black hole
        'mass_2': m2z,  # Detector-frame mass of the secondary black hole
        'a_1': a1,  # Spin of the primary black hole
        'a_2': a2,  # Spin of the secondary black hole
        'tilt_1': tilt1,  # Tilt angle of the spin of the primary black hole
        'tilt_2': tilt2,  # Tilt angle of the spin of the secondary black hole
        'ra': RA,  # Right ascension (sky position)
        'dec': dec,  # Declination (sky position)
        'theta_jn': theta_jn,  # Inclination angle
        'phi_jl': phi_jl,  # Spin-orbit precession angle
        'phi_12': phi_12,  # Spin-spin precession angle
        'psi': psi,  # Polarization angle
        'geocent_time': geo_time  # Geocentric time of the event
    }
    
    # Create a pandas DataFrame from the dictionary to hold the GW parameters for each event
    GW_data = pd.DataFrame(data_dict)
    
    # Prepare the data for the machine learning model (MLP) used to predict SNR (Signal-to-Noise Ratio)
    # The function 'prep_data_for_MLP_full' prepares the GW parameters for input into the MLP model.
    x_data_MLP = utilities.prep_data_for_MLP_full(GW_data, device)
    
    # Predict the SNR values using the MLP model
    # The model outputs SNR * distance (signal strength scaled by distance), so we divide by distance to get the actual SNR
    ypred = model.run_on_dataset(x_data_MLP.to(device))  # Run the data through the MLP model
    snr_pred = ypred.cpu().numpy() / np.array(GW_data['luminosity_distance'])[:, None]  # Normalize by distance to get SNR


    # Initialize an empty dictionary to store SNR values for each detector
    temp_dict = {}
    temp_snrs = []
    
    # Loop through the list of detectors (H1, L1, V1) and store the SNR for each detector in temp_dict
    if 'H1' in detectors:
        temp_dict['snr_h1'] = snr_pred[:, 0]  # SNR for H1
        temp_snrs.append(snr_pred[:, 0])  # Append H1 SNR to list
    
    if 'L1' in detectors:
        temp_dict['snr_l1'] = snr_pred[:, 1]  # SNR for L1
        temp_snrs.append(snr_pred[:, 1])  # Append L1 SNR to list
    
    if 'V1' in detectors:
        temp_dict['snr_v1'] = snr_pred[:, 2]  # SNR for V1
        temp_snrs.append(snr_pred[:, 2])  # Append V1 SNR to list
    
    # Calculate the network SNR by summing the squares of individual SNRs across detectors
    # Network SNR is the combined signal strength from all detectors in the network (H1, L1, V1)
    network_snr_sq = np.sum((np.array(temp_snrs) ** 2).T, axis=1)
    
    # Simulate the observed SNR by sampling from a non-central chi-squared distribution
    # The non-centrality parameter is the network SNR squared, which gives a more realistic model of the observed SNR (matched filter SNR)
    snrs_obs = np.sqrt(ncx2.rvs(2 * n_det, network_snr_sq, size=nxN, loc=0, scale=1))
    
    # Add the observed SNRs to the dictionary and create a DataFrame from the SNR values
    temp_dict['observed'] = snrs_obs
    df_temp_snrs = pd.DataFrame(temp_dict)
    
    # If a single detector SNR threshold (SNRth_single) is provided, apply it to filter individual detector events
    if SNRth_single > 0:
        # For H1 and L1 detectors, apply threshold and filter events that meet the condition
        if 'H1' and 'L1' in detectors:
            bad_inx = np.where(((df_temp_snrs['snr_h1'] > SNRth_single) &
                                (df_temp_snrs['snr_l1'] > SNRth_single) &
                                (df_temp_snrs['observed'] > SNRth)) == False)[0]
            df_temp_snrs.loc[bad_inx] = np.nan  # Mark filtered events as NaN
            GW_data['H1'] = np.array(df_temp_snrs.snr_h1)
            GW_data['L1'] = np.array(df_temp_snrs.snr_l1)
    
        # For H1 and V1 detectors
        if 'H1' and 'V1' in detectors:
            bad_inx = np.where(((df_temp_snrs['snr_h1'] > SNRth_single) &
                                (df_temp_snrs['snr_v1'] > SNRth_single) &
                                (df_temp_snrs['observed'] > SNRth)) == False)[0]
            df_temp_snrs.loc[bad_inx] = np.nan
            GW_data['H1'] = np.array(df_temp_snrs.snr_h1)
            GW_data['V1'] = np.array(df_temp_snrs.snr_v1)
    
        # For L1 and V1 detectors
        if 'L1' and 'V1' in detectors:
            bad_inx = np.where(((df_temp_snrs['snr_l1'] > SNRth_single) &
                                (df_temp_snrs['snr_v1'] > SNRth_single) &
                                (df_temp_snrs['observed'] > SNRth)) == False)[0]
            df_temp_snrs.loc[bad_inx] = np.nan
            GW_data['L1'] = np.array(df_temp_snrs.snr_l1)
            GW_data['V1'] = np.array(df_temp_snrs.snr_v1)
    
        # For all three detectors (H1, L1, V1), apply the threshold for each
        if 'H1' and 'L1' and 'V1' in detectors:
            bad_inx = np.where(((df_temp_snrs['snr_h1'] > SNRth_single) &
                                (df_temp_snrs['snr_l1'] > SNRth_single) &
                                (df_temp_snrs['snr_v1'] > SNRth_single) &
                                (df_temp_snrs['observed'] > SNRth)) == False)[0]
            df_temp_snrs.loc[bad_inx] = np.nan
            GW_data['H1'] = np.array(df_temp_snrs.snr_h1)
            GW_data['L1'] = np.array(df_temp_snrs.snr_l1)
            GW_data['V1'] = np.array(df_temp_snrs.snr_v1)
    
        # Assign the observed network SNR after applying the filtering
        GW_data['snr'] = np.array(df_temp_snrs.observed)
    
    else:
        # If no single-detector threshold is provided, just use the observed SNR
        GW_data['snr'] = snrs_obs


    # Get the indices of detected events where the observed network SNR exceeds the threshold (SNRth)
    inx_out = np.where(GW_data.snr >= SNRth)[0]
    
    # Add the current Hubble constant values (H0) and redshift (z) to the data
    GW_data['H0'] = repeated_H0
    GW_data['z'] = z   
    GW_data['inx'] = inx_gal # Mark whether the event comes from the galaxy catalog (1 for yes, 0 for no)


    if in_out:
        GW_data['app_mag'] = app_samples  # Assign apparent magnitudes from galaxy catalog
    else:
        GW_data['app_mag'] = np.ones(len(repeated_H0))  # If no catalog is used, set app_mag to 1 (placeholder)

    # For testing data, add the random numbers used for each H0 sample
    if type_of_data == 'testing':
        GW_data['R'] = repeated_Rnums
    
    # If Nselect > 1, we check events in batches and select one event per batch
    if Nselect > 1:
        inds_to_keep = []  # Initialize an empty list to store the indices of selected events
    
        # Loop over all H0 samples (n)
        for k in range(n):
            try:
                # For each batch of Nselect events, select one index where the event meets the SNR threshold
                inds_to_keep.append(inx_out[(k * int(select) < inx_out) & (inx_out < (k + 1) * int(select))][0])
            except IndexError:
                # If no event meets the threshold in the current batch, skip to the next
                pass
    
        # If no event was selected in any of the batches, increment the counter and continue to the next iteration
        if len(inds_to_keep) == 0:
            counter += 1
            continue
    else:
        # If Nselect == 1, just keep all detected events
        inds_to_keep = inx_out
    
    # Store the selected events by keeping only the indices we chose
    out_data = GW_data.loc[np.array(inds_to_keep)]
    list_data.append(out_data)  # Append the selected events to the list for final storage
    
    # Increment the counter for iterations
    counter += 1

    # If we're generating training data, handle missed H0 values
    if type_of_data == 'training':
        # Identify H0 values that were not detected in the current batch
        temp_missed_H0 = np.setxor1d(out_data['H0'].to_numpy(), repeated_H0)  # Get H0s that are not in the selected data
        
        # Find which H0s are missing and extract them
        new_missed_H0 = missed_H0[np.where(np.in1d(missed_H0, temp_missed_H0) == True)[0]]
        
        # Sort the missed H0s for the next iteration
        inx_new_missed = np.where(np.in1d(missed_H0, new_missed_H0) == True)
        new_missed_H0 = missed_H0[inx_new_missed]
        inx_missed_H0 = np.argsort(new_missed_H0)  # Sort indices of missed H0s
    
        # Update missed H0 values and their corresponding CDFs for the next round
        missed_H0 = new_missed_H0[inx_missed_H0]
        missed_cdfs_zmax = [missed_cdfs_zmax[index] for index in inx_missed_H0]
        
        # Print the status of the process (how many H0s are left to detect)
        sys.stdout.write('\rH0 we missed: {} | Nselect = {} | counter = {}'.format(len(missed_H0), nxN, counter - 1))
        
        # Update variables for the next iteration
        N_missed = len(missed_H0)
        counter_list.append(counter)
        Nmissed_list.append(N_missed)
        end = time.time()
        timer_list.append(abs(end - start))
        
        # If all H0 values are detected, break out of the loop
        if len(missed_H0) == 0:
            break  # No missed H0 values, stop the loop
    
    # If we're generating testing data, handle missed random numbers
    elif type_of_data == 'testing':
        # Identify missed random numbers (R) used in testing
        missed_R = np.setxor1d(out_data['R'].to_numpy(), repeated_Rnums)
    
        # Track which R values have been missed
        temp = []
        for x in R_nums:
            temp.append(np.where(missed_R == x)[0])
        indices_R = np.concatenate(temp, axis=0)
    
        # Update missed H0 values for the next iteration based on the missed random numbers
        missed_H0 = missed_H0[indices_R]
    
        # Print the status of the process
        sys.stdout.write('\rH0 we missed: {} | Nselect = {} | counter = {}'.format(len(missed_H0), nxN, counter - 1))
        
        # If all H0 values are detected, break out of the loop
        N_missed = len(missed_H0)
        if N_missed == 0:
            break  # No missed H0 values, stop the loop
    
print('\nFINISHED Sampling events')  # All H0 values have been processed


# Combine all the selected data from each iteration into a single DataFrame
GW_data = pd.concat(list_data)

# Define the columns to keep for the output file
output_df = GW_data[['snr', 'H0', 'luminosity_distance', 'mass_1', 'mass_2', 'ra', 'dec',
                     'a_1', 'a_2', 'tilt_1', 'tilt_2', 'theta_jn', 'phi_jl', 'phi_12', 
                     'psi', 'geocent_time', 'app_mag', 'inx', 'z']]

# Save the data to a CSV file in the specified path
detector_str = '_'.join(detectors)  # Join the list of detectors into a single string
output_df.to_csv(path_data + '/run_{}_det_{}_name_{}_catalog_{}_band_{}_batch_{}_N_{}_SNR_{}_Nelect_{}_Full_para_v1.csv'.format(
    run, detector_str, Name, in_out, band, int(batch), int(N), int(SNRth), int(Nselect)))


# If save_timer is enabled (set to 1), save the timing information for each iteration
if save_timer == 1:
    timer_data = [counter_list, Nmissed_list, timer_list]  # Store counter, missed H0s, and timer information
    # Save the timing data to a text file
    np.savetxt(path_data + 'TIMER_run_{}_det_{}_name_{}_catalog_{}_band_{}_batch_{}_N_{}_SNR_{}_Nelect_{}_Full_para_v1.txt'.format(
        run, detectors, Name, in_out, band, int(batch), int(N), int(SNRth), int(Nselect)), timer_data, delimiter=',')
    
print('\nAll data successfully saved!')




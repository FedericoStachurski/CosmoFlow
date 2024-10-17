# Standard library imports
import os  # Provides functions for interacting with the operating system, such as creating directories and handling file paths
import sys  # Access to system-specific parameters and functions, used here for modifying the module search path
import json  # Enables reading and writing JSON files for data serialization
import argparse  # Provides a way to handle command-line arguments
import multiprocessing  # Allows for creating and managing multiple processes for parallel computations

# Third-party imports
import numpy as np  # Provides numerical operations and handling of large, multi-dimensional arrays
import pandas as pd  # Used for data manipulation and analysis, especially working with tabular data
import h5py  # For interacting with HDF5 files, commonly used for storing large datasets such as gravitational wave data
import torch  # PyTorch library for tensor operations and machine learning models
import matplotlib.pyplot as plt  # Used for creating visualizations and plotting graphs
import corner  # Generates corner plots to visualize multi-dimensional data distributions
from tqdm import tqdm  # Displays progress bars in loops, useful for monitoring processing times
from scipy import interpolate  # Provides interpolation tools for numerical data points
from scipy.stats import ncx2  # Implements the non-central chi-squared distribution, often used in statistical analysis
from scipy.spatial.distance import jensenshannon  # Computes Jensen-Shannon distance to compare probability distributions
from astropy import cosmology  # Provides cosmological calculations and constants for modeling the universe
from astropy.coordinates import spherical_to_cartesian, cartesian_to_spherical  # Functions for converting between spherical and cartesian coordinate systems
sys.path.append("..")
from gw_functions import pdet_theta  # Custom module containing gravitational wave functions for detection calculations
from gw_functions.gw_SNR_v2 import run_bilby_sim  # Runs bilby simulations to calculate the signal-to-noise ratio (SNR)
from gw_functions.pdet_theta import LikelihoodDenomiantor  # Class for calculating the likelihood denominator in Bayesian inference
from train_flow.handle_flow_new_v1 import HandleFlow # Handler class for managing trained normalizing flows in cosmology

# Function to create directories if they do not exist
def create_directories(base_folder, subfolders):
    """Create base and subdirectories if they don't exist."""
    # Check if the base directory exists; if not, create it along with specified subdirectories
    if not os.path.exists(base_folder):
        os.mkdir(base_folder)  # Create the base directory
        for subfolder in subfolders:
            os.mkdir(os.path.join(base_folder, subfolder))  # Create each specified subdirectory within the base folder

# Function to load data from GWTC event
def load_data_GWTC(event, population):
    print(event)
    if int(event[2:8]) <= 190930:
        path_gw = '/data/wiay/federico/PhD/GWTC_2.1/'
        file_name = path_gw+'IGWN-GWTC2p1-v2-{}_PEDataRelease_mixed_nocosmo.h5'.format(event)
    else:   
        path_gw = '/data/wiay/federico/PhD/GWTC_3/'
        file_name = path_gw+'IGWN-GWTC3p0-v1-{}_PEDataRelease_mixed_nocosmo.h5'.format(event)

    d = h5py.File(file_name,'r')
    if population == 'BBH':
        samples = np.array(d.get('C01:IMRPhenomXPHM/posterior_samples'))
    elif population == 'NSBH':
        if event == 'GW200115_042309':
            samples = np.array(d.get('C01:IMRPhenomNSBH:LowSpin/posterior_samples'))
        elif event == 'GW190814_211039':
            samples = np.array(d.get('C01:IMRPhenomXPHM/posterior_samples'))
        else: 
            samples = np.array(d.get('C01:IMRPhenomNSBH/posterior_samples'))
    d.close()
    df = pd.DataFrame(samples)
    df = df[['luminosity_distance', 'ra', 'dec', 'mass_1', 'mass_2',
                 'a_1', 'a_2','tilt_1', 'tilt_2', 'theta_jn', 'phi_jl',
                 'phi_12', 'psi','geocent_time', 'network_optimal_snr']]
    return df

    # Store the relevant columns from the posterior samples in a DataFrame and return it
    df = pd.DataFrame(samples)  # Convert the samples to a DataFrame
    return df[['luminosity_distance', 'ra', 'dec', 'mass_1', 'mass_2', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'theta_jn', 'phi_jl', 'phi_12', 'psi', 'geocent_time', 'network_optimal_snr']]  # Return the selected columns


# Parse arguments
# Set up argument parsing to collect command-line arguments for configuration options
ap = argparse.ArgumentParser()  # Create an argument parser
ap.add_argument("-Folder", "--Name_folder", required=True, help="Name of the folder to save the gravitational wave posteriors")  # Argument for the folder name to save results
ap.add_argument("-Nsamples", "--samples", required=True, help="Number of posterior samples to use")  # Argument for number of samples to use
ap.add_argument("-SNRth", "--SNRth", required=True, help="Signal-to-noise ratio (SNR) threshold")  # Argument for SNR threshold
ap.add_argument("-Flow", "--Flow", required=True, help="Trained flow model to use")  # Argument for specifying the trained flow model
ap.add_argument("-population", "--population", required=True, help="Population type, e.g., BBH or NSBH")  # Argument for population type
ap.add_argument("-run", "--run", required=True, help="Observing run to consider, e.g., O1, O2, O3")  # Argument for specifying the observing run
ap.add_argument("-det", "--detectors", required=True, help="Detectors used, e.g., HLV, HL, HV")  # Argument for specifying the detectors used
ap.add_argument("-device", "--device", required=True, default='cpu', help="Device to use for computation: CPU or CUDA-enabled GPU")  # Argument for the computation device
ap.add_argument("-epoch", "--epoch", required=False, default=None, help="Epoch number of the trained flow model to use")  # Argument for the epoch number of the flow model
args = vars(ap.parse_args())  # Parse the arguments and convert them to a dictionary

# Extract arguments
# Extract the command-line arguments to local variables for easier reference
Folder = str(args['Name_folder'])  # Folder to save results
Nsamples = int(args['samples'])  # Number of samples to use
rth = int(args['SNRth'])  # SNR threshold
Flow = str(args['Flow'])  # Trained flow model name
population = str(args['population'])  # Population type (BBH or NSBH)
run = str(args['run'])  # Observing run (O1, O2, O3)
detectors = str(args['detectors'])  # Detectors used
device = str(args['device'])  # Computation device (CPU or CUDA)
epoch = args['epoch']  # Epoch number (if provided)
if epoch is not None:
    epoch = int(args['epoch'])  # Convert epoch to integer if not None

# Get events based on run and detectors
# Retrieve the relevant events based on the specified observing run and detectors, raise an error if not found
# Organizes all the available events for each observing run to facilitate easy access
if run == 'O1':
    events = [ 'GW150914_095045','GW151226_033853'] # O1
elif run == 'O2':
    if detectors == 'HLV':
        events = ['GW170809_082821', 'GW170814_103043', 'GW170818_022509'] #O2
    elif detectors == 'HL':
        events = ['GW170104_101158', 'GW170608_020116', 'GW170823_131358'] #O2
    else: raise ValueError('detecotrs not found')
elif run == 'O3':
    if detectors == 'HLV':
        if population == 'NSBH':
            # events = ['GW190814_211039']
            events = [ 'GW200105_162426', 'GW200115_042309'] #'GW190814_211039',
        else: 
            events = ['GW190408_181802', 'GW190412_053044', 'GW190503_185404', 'GW190512_180714', 'GW190513_205428',
                      'GW190517_055101', 'GW190519_153544', 'GW190521_030229', 'GW190602_175927', 'GW190701_203306',
                      'GW190720_000836', 'GW190727_060333', 'GW190728_064510', 'GW190828_063405', 'GW190828_065509',
                      'GW190915_235702', 'GW190924_021846', 'GW200129_065458', 'GW200202_154313', 'GW200224_222234',
                      'GW200311_115853']
    elif detectors == 'HL':
        events = ['GW190521_074359', 'GW190706_222641', 'GW190707_093326', 'GW191109_010717', 'GW191129_134029',
                  'GW191204_171526','GW191222_033537', 'GW200225_060421']
    elif detectors == 'HV':
        events = ['GW191216_213338']
    elif detectors == 'LV':
        events = ['GW190630_185205', 'GW190708_232457', 'GW190910_112807', 'GW200112_155838' ]
    else: raise ValueError('detecotrs not found')


def get_likelihoods(h0, df, N_samples, flow_class):
    likelihood_vertical = [] # Initialize an empty list
    for h in tqdm(h0): #loop over h0 values
        likelihood_vertical.append(flow_class.evaluate_log_prob(df.loc[:N_samples-1], np.repeat(h,N_samples))) # append likelihood value to the empty list
        
    return  np.array(likelihood_vertical).T
    
    





# Set up cosmology and likelihood
# Set up the cosmological model and initialize the likelihood denominator class for inference
h0 = np.linspace(20, 140, 500)  # Hubble constant range for posterior estimation
cosmo_bilby = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)  # Initialize a flat Lambda-CDM cosmological model with H0 and Om0 values
# Instantiate the likelihood denominator class with the SNR threshold and cosmological model
denominator_class = LikelihoodDenomiantor(rth, cosmo_bilby, len(detectors))  # Create an instance of LikelihoodDenomiantor
path = '../train_flow/trained_flows_and_curves/'  # Path to trained flow models
flow_name = Flow  # Flow model name
flow_class = HandleFlow(path=path, flow_name=flow_name, device='cpu') # Load the trained flow model for inference

# Create necessary directories
# Create the output directories if they do not already exist
create_directories(Folder, ['plots', 'posteriors'])  # Create directories for saving results

def get_gwcosmo_posterior(event_name, H0vec):

    ####### With Catalogue
    #GWTC catalog short names (check if event name has a short name)
    short_names = ['GW150914', 'GW151226', 'GW170104', 'GW170608', 'GW170809', 'GW170814', 'GW170818', 'GW170823','GW190412', 'GW190521', 'GW190814']
    for name in short_names:
        if event_name == 'GW190521_074359':
            break
        elif event_name[:8] == name:
            event_name = name
            break 

    #load gwcosmo results for same event
    O3_events_posteriors = json.load(open('/data/wiay/federico/PhD/O3_Posteriors_file/O3_gwcosmo_H0_event_posteriors.json'))
    posterior_of_event = O3_events_posteriors['Mu_g_32.27_Mmax_112.5_band_K_Lambda_4.59'][event_name]
    H0_grid = O3_events_posteriors['H0_grid']  
    
    ####EMPTY Catalogue
    path_empty = '/data/wiay/federico/PhD/o3-cosmology/gwcosmo_results/mature_circulation_material/results/Mu_g_32.27_Mmax_112.5_band_K_Lambda_4.59_empty/'
    if (run == 'O1') or (run == 'O2'):
        for name in short_names:
            if event_name[:8] == name:
                empty_file = name
                break
            else: 
                empty_file = event_name
    elif run == 'O3':
        data = os.listdir(path_empty) 
        event = event_name[:8]
        for file in data:
            if file.startswith(event):
                empty_file = file
  
        
    path_file_npz = path_empty + empty_file+'/'+empty_file+'.npz' 
    data_empty = np.load(path_file_npz, allow_pickle=True)
    data_empty = data_empty['arr_0']
    H0_grid_empty = data_empty[0]
    posterior_empty = data_empty[2] ## Organize the gwcosmo data into arrays

    #Interpolate Catalog
    f = interpolate.interp1d(H0_grid, posterior_of_event)
    ynew = f(H0vec) 
    post_O3 = ynew/np.sum(ynew*dH)   
    
    #Interpolate Empty
    f = interpolate.interp1d(H0_grid_empty, posterior_empty)
    ynew = f(H0vec) 
    post_O3_empty = ynew/np.sum(ynew*dH)  

    return post_O3, post_O3_empty




# Process each GW event
for GW_event in events:
    print(f'Computing event {GW_event}')  # Print the event being processed
    # Load the data for the given gravitational wave event
    df = load_data_GWTC(GW_event, population)  # Load event data into a DataFrame
    H0vec = np.linspace(20, 140, 500)  # Hubble constant values to evaluate
    SNRs = df.network_optimal_snr  # Extract network optimal SNRs from the data
    pD = denominator_class.p_D_theta(SNRs)  # Calculate detection probability for each SNR value
    pt = np.array(denominator_class.p_theta_omega_cosmo(df))  # Compute the prior distribution values for each data point
    dH = np.diff(H0vec)[0]  # Calculate the step size for Hubble constant values
    # Select specific columns of the DataFrame needed for likelihood calculations
    df_datainput = df[['luminosity_distance', 'ra', 'dec', 'mass_1', 'mass_2', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'theta_jn', 'phi_jl', 'phi_12', 'psi', 'geocent_time']]  # Extract relevant columns for likelihood calculations
    likelihoods = get_likelihoods(H0vec, df_datainput, Nsamples, flow_class)  # Obtain likelihoods for each H0 value

    # Plot and save results
    # Generate plots for the posterior distribution and save the results
    plt.figure(figsize=(10, 8))  # Create a new figure for the plot
    posterior = np.zeros(len(H0vec))  # Initialize an array to store posterior values
    
    for i, like in enumerate(likelihoods):
        # Plot individual likelihoods for visualization
        # like = like.flatten() # Flatten the likelihood for easy calculations
        like = np.exp(like)
        plt.plot(H0vec, like / np.trapz(like, x=H0vec), alpha=0.05, color='red', linewidth=1)  # Plot each likelihood, normalized
        posterior += (like / (pD[i] * pt[i]))  # Update posterior with weighted likelihood value
        if i == Nsamples:
            break  # Stop if the number of samples reaches the limit

    # Normalize the posterior distributions to ensure proper probability distributions
    posterior /= Nsamples  # Average the posterior values over the number of samples
    posterior /= np.trapz(posterior, x=H0vec)  # Normalize the posterior to ensure it integrates to 1


    # Save the plots and posterior data to the appropriate directories
    plt.title(GW_event)  # Set the title of the plot, including event name and placeholders for JS values
    plt.plot(H0vec, posterior, color='blue', linewidth=5, label='CosmoFlow')  # Plot the posterior with label

    post_O3, post_O3_empty = get_gwcosmo_posterior(GW_event, H0vec) # Get gwcosmo resutls
    
    plt.plot(H0vec, post_O3, color='black', linewidth=5, label='gwcosmo (with catalogue)')  # Plot the gwcosmo posterior with label w/ catalogue
    plt.plot(H0vec, post_O3_empty, color='green', linewidth=5, label='gwcosmo (empty catalogue)')  # Plot the posterior with label EMPTY catalogue
    plt.ylim([0.00, 0.025])  # Set the limits for the y-axis
    plt.xlim([20, 140])  # Set the limits for the x-axis
    plt.legend(loc='best', fontsize=15)  # Add a legend to the plot
    plt.grid(True, alpha=0.5)  # Add a grid to the plot with some transparency
    plt.xlabel(r'$H_{0} \: [km \: s^{-1} \: Mpc^{-1}]$', fontsize=25)  # Set the x-axis label with proper formatting
    plt.ylabel(r'$p(H_{0}) \: [km^{-1} \: s \: Mpc]$', fontsize=25)  # Set the y-axis label with proper formatting
    plt.savefig(f'{Folder}/plots/{GW_event}')  # Save the plot to the specified directory
    plt.close()  # Close the figure to free up memory
    # Save the computed posterior distributions as text files
    np.savetxt(f'{Folder}/posteriors/{GW_event}.txt', posterior)  # Save the posterior data to a text file
    print('Saving Posterior')  # Print confirmation that posterior data is being saved

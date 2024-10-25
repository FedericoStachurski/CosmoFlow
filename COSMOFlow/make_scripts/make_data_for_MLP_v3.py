import os, sys
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

# Add current directory and parent directory to the system path for importing modules
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
print(parentdir)

# Import custom gravitational wave functions from parent directory
from gw_functions import gw_priors_v2
from gw_functions import gw_SNR_v2
from tqdm import tqdm 
import multiprocessing
from scipy.stats import loguniform

# Import argparse to handle command-line arguments for the script
import argparse

# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser for various configurations required by the script
ap.add_argument("-Name", "--Name_file", required=True,
   help="Name of data")
ap.add_argument("-type", "--type_data", required=True,
   help="Type of data: OPTIONS[training, testing]")
ap.add_argument("-seed", "--seed", required=False,
   help="Seed for random number generation", default=1996)
ap.add_argument("-detector", "--detector", nargs='+', required=True,
   help="Detectors to use: OPTIONS [H1, L1, V1]", default='H1')
ap.add_argument("-run", "--run", required=True,
   help="Observing run: OPTIONS [O1, O2, O3, O4]", default='O3')
ap.add_argument("-N", "--N", required=True,
   help="Number of samples in the dataset", default=100_000)
ap.add_argument("-threads", "--threads", required=False,
   help="Number of threads for multiprocessing", default=10)
ap.add_argument("-approximator", "--wave_approx", required=True,
   help="Wave approximator to use", default='IMRPhenomXPHM')
ap.add_argument("-batch", "--batch", required=True,
   help="Batch number for data generation", default=1)

# Parse the arguments
args = vars(ap.parse_args())

# Extracting the command-line arguments into variables for easier reference
Name = str(args['Name_file'])
type_of_data = str(args['type_data'])
N = int(args['N'])
threads = int(args['threads'])
seed = int(args['seed'])
det = args['detector']
run = str(args['run'])
threads = int(args['threads'])
batch = int(args['batch'])
approximator = str(args['wave_approx'])

print(det)
np.random.seed(seed)  # Set seed for reproducibility

# Set data type (either 'training' or 'testing')
type_data = type_of_data

# Number of samples to generate
N = N
snr = []  # Placeholder for storing signal-to-noise ratio (SNR) values

# Draw prior samples using the custom function `gw_priors_v2`
_, _, _, a1sample, a2sample, tilt1sample, tilt2sample, RAsample, decsample, theta_jnsample, phi_jlsample, phi_12sample, psisample, phasesample, geo_time = gw_priors_v2.draw_prior(N)

# Generate random samples for luminosity distance and masses
dlsample = np.random.uniform(10, 11_000, N)  # Luminosity distance samples
m1zsample = np.random.uniform(4, 350, N)  # Mass 1 (primary object)
m2zsample = np.random.uniform(0.5, 15, N)  # Mass 2 (secondary object for NSBH)

# Swap mass values if m1 is less than m2 to ensure m1 >= m2
inx = np.where(m1zsample < m2zsample)[0]
temp_m1 = m1zsample[inx]
temp_m2 = m2zsample[inx]
m1zsample[inx] = temp_m2
m2zsample[inx] = temp_m1

# Create a dictionary to store the sampled data
data = {
    'luminosity_distance': dlsample, 'mass_1': m1zsample, 'mass_2': m2zsample, 'a_1': a1sample, 'a_2': a2sample, 
    'tilt_1': tilt1sample, 'tilt_2': tilt2sample, 'ra': RAsample, 'dec': decsample, 'theta_jn': theta_jnsample, 
    'phi_jl': phi_jlsample, 'phi_12': phi_12sample, 'psi': psisample, 'phase': 0, 'geocent_time': geo_time
}

# Convert the data dictionary into a Pandas DataFrame
df = pd.DataFrame(data)

# Define a function to compute SNR using multiprocessing
def compute_SNR(inx):
    return gw_SNR_v2.run_bilby_sim(df, inx, det, run, approximator, snr_type='optimal')

# Create an array of indices representing the samples to be processed
indicies = np.arange(N)
SNRs_list = []

# Use multiprocessing to compute SNR values for each sample
with multiprocessing.Pool(threads) as p:
    SNRs = list(tqdm(p.imap(compute_SNR, indicies), total=N))
SNRs = np.array(SNRs).T  # Transpose the resulting SNR array

# Update the data dictionary with the computed SNR values for each detector
data.update({'snr_H1': SNRs[0], 'snr_L1': SNRs[1], 'snr_V1': SNRs[2]})
df = pd.DataFrame(data)

# Reorganize columns for better readability
df = df[['luminosity_distance', 'mass_1', 'mass_2', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'ra',
         'dec', 'theta_jn', 'phi_jl', 'phi_12', 'psi', 'geocent_time', 'snr_H1', 'snr_L1', 'snr_V1']]

# Save the generated data to CSV files based on whether the data is for training or testing
if type_data == 'training':
    path_data = r"data_for_MLP/training/"
    if len(det) == 3:
        df.to_csv(path_data + '_{}_{}_det_{}_{}_{}_run_{}_approx_{}_batch_{}.csv'.format(Name, N, *det, run, approximator, batch))
    elif len(det) == 2:
        df.to_csv(path_data + '_{}_{}_det_{}_{}_run_{}_approx_{}_batch_{}.csv'.format(Name, N, *det, run, approximator, batch))
elif type_data == 'testing':
    path_data = r"data_for_MLP/testing/"
    if len(det) == 3:
        df.to_csv(path_data + 'testing_data_{}_det_{}_{}_{}_run_{}_batch_{}.csv'.format(N, *det, run, batch))
    elif len(det) == 2:
        df.to_csv(path_data + 'testing_data_{}_det_{}_{}_run_{}_batch_{}.csv'.format(N, *det, run, batch))

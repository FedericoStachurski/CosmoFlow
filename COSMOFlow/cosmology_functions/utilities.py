import numpy as np 
import healpy as hp
import argparse
import torch
from astropy.time import Time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from scipy.stats import norm, gaussian_kde
from scipy.stats import entropy
from astropy.coordinates import spherical_to_cartesian, cartesian_to_spherical
from torch import logit, sigmoid
import h5py

def round_base(x, base=100):
    return int(base * round(float(x)/base))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def convert_gps_sday(gps):
    return gps%86164.0905        
        
def h_samples_alpha(N, alpha, hmin=20, hmax=140):
    #alpha is alpha + 1 for H0^(alpha - 1)
    uniform_samples = np.random.uniform(0,1, size = N)
    if alpha == -1:
        return np.exp(uniform_samples*np.log(hmax) + (1-uniform_samples)*np.log(hmin))
        
    else: 
        return (uniform_samples*(hmax**(alpha+1)) + (1 - uniform_samples)*hmin**(alpha+1))**(1/(alpha+1))

def ha_from_ra_gps(gps, ra):
    return (gps/(3600*24))%(86164.0905/(3600*24))*2*np.pi - ra

def ha_from_GPS_RA(gps, ra):
    Time_class = Time(gps, format='gps')
    time = Time_class.sidereal_time('apparent', 'greenwich').value
    ha = (time - ra) * 15 * (np.pi/180)
    #ha = ha%2*np.pi
    return ha

def prep_data_for_MLP(df, device):
    data_testing = df[['mass_1','mass_2',
                        'ra','dec',
                       'theta_jn',
                       'psi','geocent_time']]
    # ha_testing = ha_from_ra_gps(np.array(data_testing.ra), np.array(data_testing.geocent_time))
    # df['ha'] = ha_testing
    # data_testing = df[['mass_1','mass_2', 'theta_jn', 'ha', 'dec', 'psi']]
    df[['geocent_time']] = df[['geocent_time']]%86164.0905
    data_testing = df[['mass_1','mass_2', 'theta_jn', 'ra', 'dec', 'psi', 'geocent_time']]
    xdata_testing = torch.as_tensor(data_testing.to_numpy(), device=device).float()
    return xdata_testing



def prep_data_for_MLP_full(df, device):
    data_testing = df[['mass_1','mass_2','theta_jn', 'ra', 'dec','psi', 'geocent_time','a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_jl', 'phi_12' ]]
    df[['geocent_time']] = df[['geocent_time']]%86164.0905
    data_testing = df[['mass_1','mass_2','theta_jn', 'ra', 'dec','psi', 'geocent_time','a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_jl', 'phi_12' ]]
    xdata_testing = torch.as_tensor(data_testing.to_numpy(), device=device).float()
    return xdata_testing



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



def logit_data(data_to_logit):
    a1_logit = logit(torch.from_numpy(np.array(data_to_logit.a1)))
    a2_logit = logit(torch.from_numpy(np.array(data_to_logit.a2)))
    phijl_logit = logit(torch.from_numpy(np.array(data_to_logit.phi_jl)))
    phi12_logit = logit(torch.from_numpy(np.array(data_to_logit.phi_12)))
    pol_logit = logit(torch.from_numpy(np.array(data_to_logit.psi)))
    tc_logit = logit(torch.from_numpy(np.array(data_to_logit.geocent_time)))

    data_to_logit.loc[:,'a1'] = np.array(a1_logit)
    data_to_logit.loc[:,'a2'] = np.array(a2_logit)
    data_to_logit.loc[:,'phi_jl'] = np.array(phijl_logit)
    data_to_logit.loc[:,'phi_12'] = np.array(phi12_logit)
    data_to_logit.loc[:,'psi'] = np.array(pol_logit)
    data_to_logit.loc[:,'geocent_time'] = np.array(tc_logit)
    return data_to_logit

def sigmoid_data(data_to_sigmoid):
    a1_sigmoid= sigmoid(torch.from_numpy(np.array(data_to_sigmoid.a1)))
    a2_sigmoid = sigmoid(torch.from_numpy(np.array(data_to_sigmoid.a2)))
    phijl_sigmoid = sigmoid(torch.from_numpy(np.array(data_to_sigmoid.phi_jl)))
    phi12_sigmoid = sigmoid(torch.from_numpy(np.array(data_to_sigmoid.phi_12)))
    pol_sigmoid = sigmoid(torch.from_numpy(np.array(data_to_sigmoid.psi)))
    tc_sigmoid = sigmoid(torch.from_numpy(np.array(data_to_sigmoid.geocent_time)))

    data_to_sigmoid.loc[:,'a1'] = np.array(a1_sigmoid)
    data_to_sigmoid.loc[:,'a2'] = np.array(a2_sigmoid)
    data_to_sigmoid.loc[:,'phi_jl'] = np.array(phijl_sigmoid)
    data_to_sigmoid.loc[:,'phi_12'] = np.array(phi12_sigmoid)
    data_to_sigmoid.loc[:,'psi'] = np.array(pol_sigmoid)
    data_to_sigmoid.loc[:,'geocent_time'] = np.array(tc_sigmoid)
    return data_to_sigmoid


def scale_data(data_to_scale, Scaler, n_conditionals = 1):
    target = data_to_scale[data_to_scale.columns[0:-n_conditionals]]
    if n_conditionals == 1:
        conditioners = np.array(data_to_scale[data_to_scale.columns[-n_conditionals]]).reshape(-1,1)
    else:
        conditioners = np.array(data_to_scale[data_to_scale.columns[-n_conditionals:]])

    if Scaler == 'MinMax':
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
    elif Scaler == 'Standard':
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
    else: 
        raise ValueError('Scaler = {} is not an option, use either Standard or MinMax'.format(Scaler))
    scaled_target = scaler_x.fit_transform(target) 
    scaled_conditioners = scaler_y.fit_transform(conditioners)  
    scaled_data = np.hstack((scaled_target, scaled_conditioners))
    scaled_data = pd.DataFrame(scaled_data, index=data_to_scale.index, columns=data_to_scale.columns)
    return scaler_x, scaler_y, scaled_data


def KL_evaluate_gaussian(samples, gaussian_samples, gauss_vector):
    def pdf_evaluate(samples):
        density = gaussian_kde(samples)
        kde_points = density.pdf(gauss_vector)
        return np.array(kde_points)
    pdf_samples = pdf_evaluate(samples)
    return pdf_samples, entropy(pdf_samples, gaussian_samples)

def load_data_GWTC(event, path = None):
    if path is None:
        path = '/data/wiay/federico/PhD'
        
    if int(event[2:8]) <= 190930:
        path_gw = path+'/GWTC_2.1/'
        file_name = path_gw+'IGWN-GWTC2p1-v2-{}_PEDataRelease_mixed_nocosmo.h5'.format(event)
    else:   
        path_gw = path+'/GWTC_3/'
        file_name = path_gw+'IGWN-GWTC3p0-v1-{}_PEDataRelease_mixed_nocosmo.h5'.format(event)
    
    d = h5py.File(file_name,'r')
    samples = np.array(d.get('C01:IMRPhenomXPHM/posterior_samples'))
    d.close()
    df = pd.DataFrame(samples)
    df = df.loc[df.mass_2 > 4.8]
    df =  df.sample(frac=1).reset_index(drop=True)
    return df




def proposal_pdf(low,high,N):
    samples_proposal = np.random.uniform(low,high,N)
    prod_result_target = np.ones(N)
    return samples_proposal, prod_result_target

def make_samples(N_samples, dimensions):
    h0_samples_proposal, _ = proposal_pdf(20,180,N_samples)
    om0_samples_proposal, _ = proposal_pdf(0,1,N_samples)
#     gamma_samples_proposal, _ = proposal_pdf(0,12,N_samples)
#     kappa_samples_proposal, _ = proposal_pdf(0,6,N_samples)
#     zp_samples_proposal, _ = proposal_pdf(0,4,N_samples)
    
#     alpha_samples_proposal, _ = proposal_pdf(1.5,12,N_samples)
#     beta_samples_proposal, _ = proposal_pdf(-4.0,12,N_samples)
#     mmax_samples_proposal, _ = proposal_pdf(50.0,200.0,N_samples)
#     mmin_samples_proposal, _ = proposal_pdf(2.0,10.0,N_samples)
    
#     mug_samples_proposal, _ = proposal_pdf(20.0,50.0,N_samples)
#     sigmag_samples_proposal, _ = proposal_pdf(0.4,10.0,N_samples)
#     lambda_samples_proposal, _ = proposal_pdf(0.0,1.0,N_samples)
#     delta_samples_proposal, _ = proposal_pdf(0.0,10.0,N_samples)
    
    # proposed_samples = [h0_samples_proposal, gamma_samples_proposal, kappa_samples_proposal, zp_samples_proposal, alpha_samples_proposal,
    #                    beta_samples_proposal, mmax_samples_proposal, mmin_samples_proposal, mug_samples_proposal, sigmag_samples_proposal,
    #                    lambda_samples_proposal, delta_samples_proposal]
    # proposed_samples = np.array(proposed_samples).reshape(12,N_samples)
    # proposed_samples = [h0_samples_proposal, gamma_samples_proposal, mmax_samples_proposal, mug_samples_proposal]
    proposed_samples = [h0_samples_proposal, om0_samples_proposal]
    
    proposed_samples = np.array(proposed_samples).reshape(dimensions,N_samples)
    
    return proposed_samples

def prior_samples(N_samples, parameters_dictionary):
    dimensions = len(parameters_dictionary)
    prior_samples = []
    if 'H0' in parameters_dictionary:
        samples_proposal, _ = proposal_pdf(parameters_dictionary['H0'][0],parameters_dictionary['H0'][1],N_samples)
        prior_samples.append(samples_proposal)
        
    if 'gamma' in parameters_dictionary:
        samples_proposal, _ = proposal_pdf(parameters_dictionary['gamma'][0],parameters_dictionary['gamma'][1],N_samples)
        prior_samples.append(samples_proposal)
        
    if 'k' in parameters_dictionary:
        samples_proposal, _ = proposal_pdf(parameters_dictionary['k'][0],parameters_dictionary['k'][1],N_samples)
        prior_samples.append(samples_proposal)
        
    if 'zp' in parameters_dictionary:
        samples_proposal, _ = proposal_pdf(parameters_dictionary['zp'][0],parameters_dictionary['zp'][1],N_samples)
        prior_samples.append(samples_proposal)
        
    if 'Om0' in parameters_dictionary:
        samples_proposal, _ = proposal_pdf(parameters_dictionary['Om0'][0],parameters_dictionary['Om0'][1],N_samples)
        prior_samples.append(samples_proposal)
        
    if 'Ktest' in parameters_dictionary:
        samples_proposal, _ = proposal_pdf(parameters_dictionary['Ktest'][0],parameters_dictionary['Ktest'][1],N_samples)
        prior_samples.append(samples_proposal)
        
        
    prior_samples = np.array(prior_samples).reshape(dimensions,N_samples)
    
    return prior_samples
        
    
    
    
    
    
    
#     om0_samples_proposal, _ = proposal_pdf(0,1,N_samples)
#     gamma_samples_proposal, _ = proposal_pdf(0,12,N_samples)
#     kappa_samples_proposal, _ = proposal_pdf(0,6,N_samples)
#     zp_samples_proposal, _ = proposal_pdf(0,4,N_samples)
    
#     alpha_samples_proposal, _ = proposal_pdf(1.5,12,N_samples)
#     beta_samples_proposal, _ = proposal_pdf(-4.0,12,N_samples)
#     mmax_samples_proposal, _ = proposal_pdf(50.0,200.0,N_samples)
#     mmin_samples_proposal, _ = proposal_pdf(2.0,10.0,N_samples)
    
#     mug_samples_proposal, _ = proposal_pdf(20.0,50.0,N_samples)
#     sigmag_samples_proposal, _ = proposal_pdf(0.4,10.0,N_samples)
#     lambda_samples_proposal, _ = proposal_pdf(0.0,1.0,N_samples)
#     delta_samples_proposal, _ = proposal_pdf(0.0,10.0,N_samples)
    
    # proposed_samples = [h0_samples_proposal, gamma_samples_proposal, kappa_samples_proposal, zp_samples_proposal, alpha_samples_proposal,
    #                    beta_samples_proposal, mmax_samples_proposal, mmin_samples_proposal, mug_samples_proposal, sigmag_samples_proposal,
    #                    lambda_samples_proposal, delta_samples_proposal]
    # proposed_samples = np.array(proposed_samples).reshape(12,N_samples)
    # proposed_samples = [h0_samples_proposal, gamma_samples_proposal, mmax_samples_proposal, mug_samples_proposal]




def split_into_four(number, number_split):
    # Calculate the closest integer division by 4
    quotient = number // int(number_split)
    remainder = number % int(number_split)
    # Initialize the list to store the four integers
    result = [quotient] * int(number_split)
    # Distribute the remainder to the first few elements
    for i in range(remainder):
        result[i] += 1
    return result

def random_det_setup(run, N):
    if run == 'O2' or run == 'O3':
        det_setup = [[1,1,1],[1,1,0], [1,0,1],[0,1,1]]
        N_setups = split_into_four(N, len(det_setup))
        HLV = np.repeat([1,1,1], repeats=N_setups[0], axis=0).reshape(3,N_setups[0]).T
        HL = np.repeat([1,1,0], repeats=N_setups[1], axis=0).reshape(3,N_setups[1]).T
        HV = np.repeat([1,0,1], repeats=N_setups[2], axis=0).reshape(3,N_setups[2]).T
        LV = np.repeat([0,1,1], repeats=N_setups[3], axis=0).reshape(3,N_setups[3]).T
        
        result = np.concatenate((HLV,HL, HV, LV))
        np.random.shuffle(result)
    elif run == 'O1':
        result = np.repeat([1,1,0], repeats=N, axis=0).reshape(3,N).T
        
    return result

def replace_small_values(matrix, threshold=1e-5):
    matrix[np.abs(matrix) < threshold] = 0
    return matrix

def make_det_setup_dataframe(setup, N):
    def zeros_or_ones(det):
        if det in setup:
            return np.ones(N)
        else: 
            return np.zeros(N)
    h1 = zeros_or_ones('H1')
    l1 = zeros_or_ones('L1')
    v1 = zeros_or_ones('V1')
    dictionary_setup = {'H1': h1, 'L1': l1, 'V1': v1}
    return pd.DataFrame(dictionary_setup)


def _MLP_luminosity_distance(z_samples,H0_samples,Om0_samples, model, device = 'cpu'):
    testing_data = {'z':z_samples, 'H0':H0_samples, 'Om0':Om0_samples}
    testing_data = pd.DataFrame(testing_data)
    data_testing = testing_data[['z','Om0']]
    xdata_testing = torch.as_tensor(data_testing.to_numpy(), device=device).float()
    ypred = model.run_on_dataset(xdata_testing.to(device))
    DL_pred = ypred.cpu().numpy()/np.array(testing_data.H0)
    return DL_pred





def upscale_map(map,NSIDE_low,NSIDE_high):
    "Upscale in resolution the magnitude threhold map"

    #Get number of pixels in high res
    Npix_high = hp.nside2npix(NSIDE_high)

    # Apply this logic to all pixels in your high-resolution catalog
    associated_mth_values = []

    for i in range(Npix_high):
        # Calculate the corresponding low NSIDE pixel for the current high NSIDE pixel
        low_res_pixel = hp.pixelfunc.nest2ring(NSIDE_low, hp.pixelfunc.ring2nest(NSIDE_high, i) // (NSIDE_high // NSIDE_low)**2)
        # Retrieve the magnitude threshold for this low_res_pixel
        mag_threshold = map[low_res_pixel]
        associated_mth_values.append(mag_threshold)
    associated_mth_values = np.array(associated_mth_values)
    return associated_mth_values
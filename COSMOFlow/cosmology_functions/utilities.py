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
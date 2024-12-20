import pandas as pd
import astropy.units as u
from astropy.coordinates import Distance
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ncx2
from tqdm import tqdm
import bilby
from scipy.interpolate import splrep, splev
from bilby.core.prior import Uniform, Sine, Constraint, Cosine
from scipy.integrate import  quad
import numpy.random as rn
import sys
import bilby.gw.utils as ut



bilby.core.utils.log.setup_logger(log_level=0)



#  Define injection parameters
def SNR_from_inj( dl, m1_det, m2_det, a1, a2, tilt1, tilt2, RA, dec, theta_jn, phi_jl, phi_12, psi, phase, geo_time):
    "compute_SNR with all 15 plus z"
    
    #geo_time = np.random.uniform(-86400, 86400, size = 1)
    
    injection_dict=dict(mass_1=m1_det, 
                        mass_2=m2_det, 
                        luminosity_distance=dl, 
                        theta_jn=theta_jn, 
                        psi=psi, 
                        phase=phase, 
                        geocent_time= geo_time, 
                        ra=RA, 
                        dec=dec, 
                        a_1=a1, 
                        a_2=a2, 
                        phi_12=phi_12, 
                        phi_jl=phi_jl, 
                        tilt_1=tilt1, 
                        tilt_2=tilt2)



    def chi_eff(a1, a2, theta1, theta2, m1, m2):
        cos_theta1 = np.cos(theta1)
        cos_theta2 = np.cos(theta2)
        q = m2 / m1 
        return (a1*cos_theta1 + q*a2*cos_theta2)/(1+q) 

    
    
    
    #  Define time and frequency parameters
    minimum_frequency=20
    duration=np.ceil(ut.calculate_time_to_merger(minimum_frequency,m1_det,m2_det))
    if duration < 1: duration = 1 
    sampling_frequency=4096
    trigger_time=injection_dict['geocent_time']
    



    #  Setup waveform generator

    waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                              reference_frequency=20., minimum_frequency=minimum_frequency)
    waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration+1.5, sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            waveform_arguments=waveform_arguments,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters)


    #  Setup interferometers and inject signal

    ifos = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
    for ifo in ifos:
        ifo.minimum_frequency = minimum_frequency
        ifo.maximum_frequency = sampling_frequency/2.
        if ifo.name == 'V1':
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(psd_file='AdV_psd.txt') 
        else:     
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(psd_file='aLIGO_early_psd.txt') 
        

    set_strain = ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency, duration=duration+1.5,
        start_time=(trigger_time - duration))
    injected_signal = ifos.inject_signal(waveform_generator=waveform_generator, parameters=injection_dict);


    SNR_H1 = abs(ifos.meta_data['H1']['matched_filter_SNR'])
    SNR_L1 = abs(ifos.meta_data['L1']['matched_filter_SNR'])
    SNR_V1 = abs(ifos.meta_data['V1']['matched_filter_SNR'])
    
    obs_SNR_H1 = SNR_H1#np.sqrt((ncx2.rvs(4, SNR_H1**2, size=1, loc = 0, scale = 1)))
    obs_SNR_L1 = SNR_L1#np.sqrt((ncx2.rvs(4, SNR_L1**2, size=1, loc = 0, scale = 1)))
    obs_SNR_V1 = SNR_V1
    
    tot_SNR = np.sqrt((obs_SNR_H1)**2 + (obs_SNR_L1)**2 + (obs_SNR_V1)**2)
    return tot_SNR

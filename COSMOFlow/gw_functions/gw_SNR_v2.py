import bilby as bl
import numpy as np
from bilby.gw import utils as gwutils

def get_length(fmin,m1,m2):

    return gwutils.calculate_time_to_merger(frequency=fmin,mass_1=m1,mass_2=m2)
    
def run_bilby_sim(injection_parameters):
    f_min = 20
    sampling_frequency = 4096
    f_max = sampling_frequency/2
    t0 = 1242442967.46
    injection_parameters["geocent_time"] = t0+injection_parameters["geocent_time"]
    
    
    detectors = ['H1', 'L1', 'V1']
    duration = np.ceil(get_length(f_min,injection_parameters["m1_det"],injection_parameters["m2_det"]))
    if duration<1: duration=1

    injection_dict=dict(a_1=injection_parameters["a1"],
                        a_2=injection_parameters["a2"],
                        tilt_1=injection_parameters["tilt1"],
                        tilt_2=injection_parameters["tilt2"],
                        geocent_time= injection_parameters["geocent_time"],
                        mass_1=injection_parameters["m1_det"],
                        mass_2=injection_parameters["m2_det"],
                        luminosity_distance=injection_parameters["dl"],
                        theta_jn=injection_parameters["theta_jn"],
                        ra=injection_parameters["RA"],
                        dec=injection_parameters["dec"],
                        phi_12=injection_parameters["phi_12"],
                        phi_jl=injection_parameters["phi_jl"],
                        psi=injection_parameters["psi"],
                        phase=injection_parameters["phase"])


    waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',reference_frequency=f_min,minimum_frequency=f_min)
    waveform_generator = bl.gw.waveform_generator.WaveformGenerator(
        sampling_frequency=sampling_frequency, duration=duration+1.5,
        frequency_domain_source_model=bl.gw.source.lal_binary_black_hole,
        parameter_conversion=bl.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments)

    ifos_list = list(detectors)
    ifos = bl.gw.detector.InterferometerList(ifos_list)
    for ifo in ifos:
        ifo.minimum_frequency = f_min
        ifo.maximum_frequency = f_max
        if ifo.name == 'V1':
            ifo.power_spectral_density = bl.gw.detector.PowerSpectralDensity(psd_file='AdV_psd.txt') 
        else:     
            ifo.power_spectral_density = bl.gw.detector.PowerSpectralDensity(psd_file='aLIGO_early_psd.txt')

    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency, duration=duration+1.5,
        start_time=injection_parameters['geocent_time']-duration)

#     try:
    ifos.inject_signal(waveform_generator=waveform_generator,
                           parameters=injection_parameters)
#     except:
#         print("Could not inject signal with injections:"+str(injection_parameters))
#         return([0,0,0,0,0]) # no data for this event
#     #            raise ValueError("Something went wrong with the injections: "+str(injection_parameters))

    det_SNR = 0
    SNRs = []
    for ifo_string in ifos_list:
        mfSNR = np.real(ifos.meta_data[ifo_string]['matched_filter_SNR'])**2
        SNRs.append(mfSNR)
        det_SNR += mfSNR
    SNRs = np.array(SNRs)
    det_SNR = np.sqrt(det_SNR)
    return det_SNR


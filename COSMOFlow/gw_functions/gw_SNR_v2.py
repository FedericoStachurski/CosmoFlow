import bilby as bl
import numpy as np
from bilby.gw import utils as gwutils

def run_bilby_sim(df,idx_n, detector, run, approximator,snr_type = 'optimal', sampling_frequency = 4096 , f_min = 20):
        #
        def get_length(fmin,m1,m2):
            return gwutils.calculate_time_to_merger(frequency=fmin,mass_1=m1,mass_2=m2)
        
        duration = np.ceil(get_length(f_min,df.mass_1[idx_n],df.mass_2[idx_n]))
        if duration<1: duration=1
        psds = {}
        

        TOA = df.geocent_time[idx_n]
                
        if (approximator == 'IMRPhenomPv2' or approximator == 'IMRPhenomXPHM' or approximator == 'IMRPhenomNSBH') is False:
            raise ValueError(' {} Approximator not being used'.format(approximator))

        detectors = detector
        run = run
            
        
        data_path = '/data/wiay/federico/PhD/cosmoflow_review/COSMOFlow/psd_data/'
        asds={'H1':{},'L1':{},'V1':{}}
        for det in detectors:
            data = np.genfromtxt(data_path+det+'_'+run+'_strain.txt')
            asds[det]['frequency']=data[:,0]
            asds[det]['psd']=data[:,1]**2
            psds[det] = bl.gw.detector.PowerSpectralDensity(frequency_array=asds[det]['frequency'],
                                                            psd_array=asds[det]['psd'])

        injection_parameters = dict(a_1=df.a_1[idx_n],
                                    a_2=df.a_2[idx_n],
                                    tilt_1=df.tilt_1[idx_n],
                                    tilt_2=df.tilt_2[idx_n],
                                    geocent_time=df.geocent_time[idx_n],
                                    mass_1=df.mass_1[idx_n],
                                    mass_2=df.mass_2[idx_n],
                                    luminosity_distance=df.luminosity_distance[idx_n],
                                    ra=df.ra[idx_n],
                                    dec=df.dec[idx_n],
                                    theta_jn=df.theta_jn[idx_n],
                                    psi=df.psi[idx_n],
                                    phase=df.phase[idx_n],
                                    phi_12=df.phi_12[idx_n],
                                    phi_jl=df.phi_jl[idx_n])

        waveform_arguments = dict(waveform_approximant=approximator,reference_frequency=f_min,minimum_frequency=f_min)
        waveform_generator = bl.gw.waveform_generator.WaveformGenerator(
            sampling_frequency=sampling_frequency, duration=duration+1.5,
            frequency_domain_source_model=bl.gw.source.lal_binary_black_hole,#DOUBLE CHECK THIS 
            parameter_conversion=bl.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=waveform_arguments)

        ifos_list = detectors
        ifos = bl.gw.detector.InterferometerList(ifos_list)
        for j in range(len(ifos)):
            ifos[j].power_spectral_density = psds[ifos_list[j]] #DUBLE CHECK 
            ifos[j].minimum_frequency = f_min
        ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=sampling_frequency, duration=duration+1.5,
            start_time=injection_parameters['geocent_time']-duration)
        
        try:
            ifos.inject_signal(waveform_generator=waveform_generator,
                               parameters=injection_parameters)
        except:
            print("Could not inject signal with injections:"+str(injection_parameters))
            return 0 # no data for this event        

        det_SNR = 0
        SNRs = []
        for ifo_string in ifos_list:
            opSNR = np.real(ifos.meta_data[ifo_string]['optimal_SNR']) #THIS SHOULD BE ONLY REAL 
            mfSNR = np.abs(ifos.meta_data[ifo_string]['matched_filter_SNR'])
            
            if snr_type == 'optimal':
                SNRs.append(opSNR)
                det_SNR += opSNR**2
                
            elif snr_type == 'matched_filter':
                SNRs.append(mfSNR)
                det_SNR += mfSNR**2
                
        det_SNR = np.sqrt(det_SNR)
        SNRs.append(det_SNR)
        return SNRs


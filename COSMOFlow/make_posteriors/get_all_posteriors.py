import matplotlib.pyplot as plt
import pandas as pd
import h5py
import numpy as np
import sys
sys.path.append("..")
from gw_functions import pdet_theta 
from tqdm import tqdm
from glasflow.flows import RealNVP, CouplingNSF
import torch 
import pickle 
import corner
from astropy.coordinates import spherical_to_cartesian, cartesian_to_spherical
from torch import logit, sigmoid
import os 
import multiprocessing 
import json
from scipy.spatial.distance import jensenshannon
from scipy import interpolate
from gw_functions.gw_SNR_v2 import run_bilby_sim
from scipy.stats import ncx2
import bilby
from astropy import cosmology
from train_flow.handle_flow import Handle_Flow
import json
import argparse

import matplotlib.pyplot as plt 
from gw_functions.pdet_theta import LikelihoodDenomiantor
from astropy import cosmology 





#pass arguments 
# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-Folder", "--Name_folder", required=True,
   help="Name of the folder to save the GW_posteriors")
ap.add_argument("-Nsamples", "--samples", required=True,
   help="Posterior samples to use")
ap.add_argument("-SNRth", "--SNRth", required=True,
   help="SNR threshold")
ap.add_argument("-Flow", "--Flow", required=True,
   help="Trained flow to use")
ap.add_argument("-population", "--population", required=True,
   help="Trained flow to use")
ap.add_argument("-run", "--run", required=True,
   help="Trained flow to use")
ap.add_argument("-det", "--detectors", required=True,
   help="detectors used [e.g. HLV, HL ... ]")
ap.add_argument("-device", "--device", required=True, default = 'cpu',
   help="device to use: cpu or cuda")
ap.add_argument("-epoch", "--epoch", required=False, default = None,
   help="which epoch nubmer to use")




args = vars(ap.parse_args())
Folder = str(args['Name_folder'])
Nsamples= int(args['samples'])
rth= int(args['SNRth'])
Flow= str(args['Flow'])
population= str(args['population'])
run = str(args['run'])
detectors= str(args['detectors'])
device= str(args['device'])
ndet = len(detectors)
print('Detectors used: '+str(ndet))
epoch = args['epoch']
if epoch is not None:
    epoch= int(args['epoch'])
h0 = np.linspace(20,140,500)

cosmo_bilby = cosmology.FlatLambdaCDM(H0 = 70, Om0 = 0.3)
denominator_class = LikelihoodDenomiantor(rth, cosmo_bilby, ndet)
path = '../train_flow/trained_flows_and_curves/'
flow_name = Flow
flow_class = Handle_Flow(path, flow_name, device, epoch = epoch)

#check if directory exists
if os.path.exists(Folder) is False:
    #Save model in folder
    os.mkdir(Folder)
    os.mkdir(Folder+'/plots')
    os.mkdir(Folder+'/posteriors')
    os.mkdir(Folder+'/posteriors_no_w')
    os.mkdir(Folder+'/O3_H0_post')
    os.mkdir(Folder+'/JS_means')

#'GW190425', 'GW190814','GW200115_042309','GW200105_162426' ??????

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
            events = ['GW190814_211039']
            # events = [ 'GW200105_162426', 'GW200115_042309'] #'GW190814_211039',
        else: 
            # events = ['GW190814_211039']
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

def load_data_GWTC(event, xyz = 0 ):
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
    print(population)
    df = pd.DataFrame(samples)
    return df


def convert_data(df):
    data = df
    def spherical_to_cart(dl, ra, dec):

        x,y,z = spherical_to_cartesian(dl, dec, ra)
        return x,y,z

    coordinates= data[['luminosity_distance', 'ra', 'dec']]
    dl = np.array(coordinates.luminosity_distance)
    ra = np.array(coordinates.ra)
    dec = np.array(coordinates.dec)
    if xyz == 0:
        return data[['luminosity_distance', 'ra', 'dec','mass_1', 'mass_2']]
    if xyz == 1:
        return data[['xcoord', 'ycoord', 'zcoord','mass_1', 'mass_2']]

def get_likelihoods(h0, df, N_samples, flow_class):
    likelihood_vertical = []
    for h in tqdm(h0):
        likelihood_vertical.append(flow_class.p_theta_H0_full_single(df.loc[:N_samples], h))
    return  np.array(likelihood_vertical).T
    
    
def get_gwcosmo_posterior(event_name, H0vec):
    
    #Catalog
    short_names = ['GW150914', 'GW151226', 'GW170104', 'GW170608', 'GW170809', 'GW170814', 'GW170818', 'GW170823','GW190412', 'GW190521', 'GW190814']
    for name in short_names:
        if event_name == 'GW190521_074359':
            break
        
        elif event_name[:8] == name:
            event_name = name
            break

   
    O3_events_posteriors = json.load(open('/data/wiay/federico/PhD/O3_Posteriors_file/O3_gwcosmo_H0_event_posteriors.json'))
    posterior_of_event = O3_events_posteriors['Mu_g_32.27_Mmax_112.5_band_K_Lambda_4.59'][event_name]
    H0_grid = O3_events_posteriors['H0_grid']  
    
    ####EMPTY
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
    posterior_empty = data_empty[2]

    #Interpolate Catalog
    f = interpolate.interp1d(H0_grid, posterior_of_event)
    ynew = f(H0vec) 
    post_O3 = ynew/np.sum(ynew*dH)   
    
    #Interpolate Empty
    f = interpolate.interp1d(H0_grid_empty, posterior_empty)
    ynew = f(H0vec) 
    post_O3_empty = ynew/np.sum(ynew*dH)  
    
    
    return post_O3, post_O3_empty


if run == 'O1':
    detectors = ['H1', 'L1']
elif (run == 'O2') or (run == 'O3') or (run == 'O4'):
    detectors = ['H1', 'L1', 'V1']





for GW_event in events:
    print('Computing event {}'.format(GW_event))
    df = load_data_GWTC(GW_event)
    threads = 20
    Npoints = 500
    H0vec = np.linspace(20,140,Npoints)
    values= np.ones(Npoints)
    SNRs = df.network_optimal_snr #ADD MLP SNR CALCULATOR 
    pD = denominator_class.p_D_theta(SNRs)
    pt = np.array(denominator_class.p_theta_omega_cosmo(df))
    dH = np.diff(H0vec)[0]
    
    likelihoods = get_likelihoods(H0vec, df, Nsamples, flow_class)
    plt.figure(figsize = (10,8))
    posterior = np.zeros(len(H0vec))
    posterio_no_w = np.zeros(len(H0vec))
    for i,like in enumerate(likelihoods):
        plt.plot(H0vec, like/np.trapz(like, x = H0vec), alpha = 0.05, color = 'red', linewidth = 1)
        posterior += (like/(pD[i]*pt[i]))
        posterio_no_w += (like)
        if i == Nsamples:
            break
    posterior = posterior/Nsamples
    posterior /= np.trapz(posterior, x = H0vec)

    posterio_no_w = posterio_no_w/Nsamples
    posterio_no_w /= np.trapz(posterio_no_w, x = H0vec)
    
    post_O3, post_O3_empty = get_gwcosmo_posterior(GW_event, H0vec)
    
    
    JS_catalog = jensenshannon(post_O3, posterior)
    JS_empty = jensenshannon(post_O3_empty, posterior)
    print('Jensen-Shannon value for CATALOG {} = {}'.format(GW_event, JS_catalog))
    print('Jensen-Shannon value for EMPTY {} = {}'.format(GW_event, JS_empty))
    
    plt.title(GW_event +r'; $JS_c = {} ; JS_e = {}$'.format(round(JS_catalog,3), round(JS_empty,3)))
    plt.plot([], [], color = 'red', label = 'likelihoods')
    plt.plot(H0vec, posterior,color = 'blue',  linewidth =5, label = 'CosmoFlow')
    # plt.plot(H0vec, posterio_no_w,color = 'tab:brown', linewidth =5, label = 'CosmoFlow | Not weighted')
    plt.plot(H0vec,post_O3, '--g', alpha=1, linewidth=5, label = 'O3 GWcosmo Posterior') 
    plt.plot(H0vec,post_O3_empty, '--k', alpha=1, linewidth=5, label = 'O3 GWcosmo Posterior EMPTY') 


    plt.ylim([0.00,0.025])
    plt.xlim([20,140])
    plt.legend(loc = 'best', fontsize = 15)
    plt.grid(True, alpha = 0.5)

    plt.xlabel(r'$H_{0} \: [km \: s^{-1} \: Mpc^{-1}]$',fontsize = 25)
    plt.ylabel(r'$p(H_{0}) \: [km^{-1} \: s \: Mpc] $',fontsize = 25)
    plt.savefig(Folder+'/plots/'+GW_event)
    np.savetxt(Folder+'/posteriors/'+GW_event+'.txt',posterior)
    np.savetxt(Folder+'/posteriors_no_w/'+GW_event+'no_w.txt',posterio_no_w)
    np.savetxt(Folder+'/O3_H0_post/O3_H0_'+GW_event+'.txt',post_O3)
    print('Saving Posterior')




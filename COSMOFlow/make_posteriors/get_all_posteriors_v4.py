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

import argparse

omega_0 = [0.305, 0, 67.90]
# cosmology=cosmology.FlatLambdaCDM(name="Planck15", H0 = H0, Om0 = omega_m)


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
ap.add_argument("-run", "--run", required=True,
   help="Trained flow to use")
ap.add_argument("-det", "--detectors", required=True,
   help="number of detectors used")



args = vars(ap.parse_args())
Folder = str(args['Name_folder'])
Nsamples= int(args['samples'])
rth= int(args['SNRth'])
Flow= str(args['Flow'])
run = str(args['run'])
detectros= str(args['ndet'])
ndet = len(detectors)
print(ndet)

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
    if detectors = 'HLV':
        events = ['GW170809_082821', 'GW170814_103043', 'GW170818_022509'] #O2
    elif detectors = 'HL':
        events = ['GW170104_101158', 'GW170608_020116', 'GW170823_131358'] #O2
elif run == 'O3':
    if detectors = 'HLV':
        events = ['GW190408_181802', 'GW190412_053044', 'GW190503_185404', 'GW190512_180714', 'GW190513_205428',
                  'GW190517_055101', 'GW190519_153544', 'GW190521_030229', 'GW190602_175927', 'GW190701_203306',
                  'GW190720_000836', 'GW190727_060333', 'GW190728_064510', 'GW190828_063405', 'GW190828_065509',
                  'GW190915_235702', 'GW190924_021846', 'GW200129_065458', 'GW200202_154313', 'GW200224_222234',
                  'GW200311_115853']
    elif detectors = 'HL':
        events = ['GW190521_074359', 'GW190706_222641', 'GW190707_093326', 'GW191109_010717', 'GW191204_171526',
                  'GW191222_033537', 'GW200225_060421']
    elif detectors = 'HV':
        events = ['GW191216_213338']
    elif detectors = 'LV':
        events = ['GW190630_185205', 'GW190708_232457', 'GW190910_112807', 'GW200112_155838', ]
    
    
    # events = ['GW190408_181802', 'GW190412_053044', 'GW190503_185404', 'GW190512_180714', 'GW190513_205428', 'GW190517_055101',
    #           'GW190519_153544', 'GW190521_030229', 'GW190602_175927', 'GW190701_203306', 'GW190720_000836', 'GW190727_060333','GW190728_064510', 
    #          'GW190828_063405', 'GW190828_065509', 'GW190910_112807', 'GW190915_235702',  
    #          'GW190924_021846', 'GW200129_065458', 
    #          'GW200202_154313', 'GW200224_222234', 'GW200311_115853']
   

def load_data_GWTC(event, xyz = 0 ):
    if int(event[2:8]) <= 190930:
        
        path_gw = '/data/wiay/federico/PhD/GWTC_2.1/'
        file_name = path_gw+'IGWN-GWTC2p1-v2-{}_PEDataRelease_mixed_nocosmo.h5'.format(event)
    else:   
        path_gw = '/data/wiay/federico/PhD/GWTC_3/'
        file_name = path_gw+'IGWN-GWTC3p0-v1-{}_PEDataRelease_mixed_nocosmo.h5'.format(event)
    
    d = h5py.File(file_name,'r')
    samples = np.array(d.get('C01:IMRPhenomXPHM/posterior_samples'))
    d.close()
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

    
def logit_data(data_to_logit):
#     a1_logit = logit(torch.from_numpy(np.array(data_to_logit.a1)))
#     a2_logit = logit(torch.from_numpy(np.array(data_to_logit.a2)))
#     phijl_logit = logit(torch.from_numpy(np.array(data_to_logit.phi_jl)))
#     phi12_logit = logit(torch.from_numpy(np.array(data_to_logit.phi_12)))
    pol_logit = logit(torch.from_numpy(np.array(data_to_logit.psi)))
    tc_logit = logit(torch.from_numpy(np.array(data_to_logit.geocent_time)))

#     data_to_logit.loc[:,'a1'] = np.array(a1_logit)
#     data_to_logit.loc[:,'a2'] = np.array(a2_logit)
#     data_to_logit.loc[:,'phi_jl'] = np.array(phijl_logit)
#     data_to_logit.loc[:,'phi_12'] = np.array(phi12_logit)
    data_to_logit.loc[:,'psi'] = np.array(pol_logit)
    data_to_logit.loc[:,'geocent_time'] = np.array(tc_logit)
    return data_to_logit

def sigmoid_data(data_to_sigmoid):
#     a1_sigmoid= sigmoid(torch.from_numpy(np.array(data_to_sigmoid.a1)))
#     a2_sigmoid = sigmoid(torch.from_numpy(np.array(data_to_sigmoid.a2)))
#     phijl_sigmoid = sigmoid(torch.from_numpy(np.array(data_to_sigmoid.phi_jl)))
#     phi12_sigmoid = sigmoid(torch.from_numpy(np.array(data_to_sigmoid.phi_12)))
    pol_sigmoid = sigmoid(torch.from_numpy(np.array(data_to_sigmoid.psi)))
    tc_sigmoid = sigmoid(torch.from_numpy(np.array(data_to_sigmoid.geocent_time)))

#     data_to_sigmoid.loc[:,'a1'] = np.array(a1_sigmoid)
#     data_to_sigmoid.loc[:,'a2'] = np.array(a2_sigmoid)
#     data_to_sigmoid.loc[:,'phi_jl'] = np.array(phijl_sigmoid)
#     data_to_sigmoid.loc[:,'phi_12'] = np.array(phi12_sigmoid)
    data_to_sigmoid.loc[:,'psi'] = np.array(pol_sigmoid)
    data_to_sigmoid.loc[:,'geocent_time'] = np.array(tc_sigmoid)
    return data_to_sigmoid

def load_hyperparameters_scalers_flow(flow_name):
    torch.set_num_threads(1)
    
    #Open hyperparameter dictionary
    path = "/data/wiay/federico/PhD/cosmoflow/COSMOFlow/train_flow/trained_flows_and_curves/"+flow_name+"/"
    hyper = open(path+'hyperparameters.txt', 'r').read()
    hyperparameters = eval(hyper)
    
    device = 'cpu'
    n_inputs = hyperparameters['n_inputs']
    n_conditional_inputs = hyperparameters['n_conditional_inputs'] 
    n_neurons = hyperparameters['n_neurons']
    n_transforms = hyperparameters['n_transforms']
    n_blocks_per_transform = hyperparameters['n_blocks_per_transform']
    dropout = hyperparameters['dropout']
    flow_type = hyperparameters['flow_type']
    
    #open scaler_x and scaler_y
    scalerfile_x = path+'scaler_x.sav'
    scalerfile_y = path+'scaler_y.sav'
    scaler_x = pickle.load(open(scalerfile_x, 'rb'))
    scaler_y = pickle.load(open(scalerfile_y, 'rb'))
  

    #Open flow model file flow.pt
    flow_load = torch.load( path+'flow.pt', map_location=device)

    if flow_type == 'RealNVP':
        flow_empty = RealNVP(n_inputs= n_inputs,
            n_transforms= n_transforms,
            n_neurons= n_neurons,
            n_conditional_inputs = n_conditional_inputs,
            n_blocks_per_transform = n_blocks_per_transform,
            batch_norm_between_transforms=True,
            dropout_probability=dropout,
            linear_transform='lu')
    elif flow_type == 'CouplingNSF':   
            flow_empty = CouplingNSF(n_inputs= n_inputs,
            n_transforms= n_transforms,
            n_neurons= n_neurons,
            n_conditional_inputs = n_conditional_inputs,
            n_blocks_per_transform = n_blocks_per_transform,
            batch_norm_between_transforms=True,
            dropout_probability=dropout,
            linear_transform='lu')
    
    flow_empty.load_state_dict(flow_load)
    flow = flow_empty
    flow.eval()
    
    return flow, hyperparameters, scaler_x, scaler_y

def pdet_rth(x, ndet = 2, rth = rth):
    p = ncx2.sf(rth**(2), 2*ndet, x**2, loc = 0, scale = 1)
    return p

def p_theta_omega(theta, omega_0 = [0.3065, 0, 67.9], cosmo_use = False):
#    "Description: Probability of GW parameters, p(theta|Omega_0) "
#    "Input: data frame of parameters"
#    "Output: p(theta) "

    omega_m, omega_k, H0 = omega_0
    cosmo=cosmology.FlatLambdaCDM(name="Planck15", H0 = H0, Om0 = omega_m)
    dl = np.array(theta.luminosity_distance)
    #Following the GWTC-2.1 GW transient catalog, the parameters are all uniforms except for dl.
    #the prior on dl is uniform in comoving volume,with flat Lambda-CDM Hubble = 67.90 and Omega_m = 0.3065
    if cosmo_use:
        p = bilby.gw.prior.UniformComovingVolume( minimum=10, maximum=20000, name = 'luminosity_distance',cosmology=cosmo)
        p = p.prob(dl)
    else: 
        p = dl**(2)*3/ (20_000**3 - 10**3)
        
    return p
    

def p_theta_H0(scaled_theta, conditional, hyper_dict):
    dCon = np.diff(conditional)[0]
    Y_H0_conditional = scaler_y.transform(conditional.reshape(-1,1))
    # theta = np.array(theta)
    # print(theta)
    samples = scaled_theta

#     dict_rand = {'x':list(samples[:,0]), 'y':list(samples[:,1]), 'z':list(samples[:,2]), 
#                   'm1':list(samples[:,3]), 'm2':list(samples[:,4]),'a1':list(samples[:,5]),
#                    'a2':list(samples[:,6]), 'tilt1':list(samples[:,7]), 'tilt2':list(samples[:,8]),
#                   'theta_jn':list(samples[:,9]), 'phi_jl':list(samples[:,10]), 'phi_12':list(samples[:,11]),
#                   'polarization':list(samples[:,12]), 'geo_time':list(samples[:,unuiform distribution in cosine prior13])}
    if hyper_dict['xyz'] == 0:
        dict_rand = {'luminosity_distance':samples[0], 'ra':samples[1], 'dec':samples[2], 'm1':samples[3], 'm2':samples[4]}
    
    
    elif hyper_dict['xyz'] == 1:
        dict_rand = {'x':samples[0], 'y':samples[1], 'z':samples[2],'m1':samples[3], 'm2':samples[4]}

    samples = pd.DataFrame(dict_rand, index = [0])
    scaled_theta = samples 
    if hyper_dict['log_it'] == 1:
        logit_data(scaled_theta)
        scaled_theta = scaled_theta[np.isfinite(scaled_theta).all(1)]

    scaled_theta = np.array(scaled_theta)

    scaled_theta = scaled_theta.T*np.ones((1,len(Y_H0_conditional)))



    conditional = np.array(Y_H0_conditional)
    data = np.array(conditional)
    data_scaled = torch.from_numpy(data.astype('float32'))

    def Flow_posterior(target, conditional, device = 'cpu'): 
        flow.eval()
        flow.to(device)

        with torch.no_grad():

            logprobx = flow.log_prob(target, conditional=conditional.to(device))
            logprobx = logprobx.numpy() 



            return  logprobx

    Log_Prob = Flow_posterior(torch.from_numpy(scaled_theta.T).float(), data_scaled)


    return np.exp(Log_Prob)#/np.sum(np.exp(Log_Prob)*dCon) 




if run == 'O1':
    detectors = ['H1', 'L1']
elif (run == 'O2') or (run == 'O3') or (run == 'O4'):
    detectors = ['H1', 'L1', 'V1']


#Get flow 
flow, hyper_dict, scaler_x, scaler_y = load_hyperparameters_scalers_flow(Flow)      
xyz = hyper_dict['xyz']
def compute_SNR(inx):
    return run_bilby_sim(df, inx, detectors, run, 'IMRPhenomXPHM')
counter = 0
tot_JS = pd.DataFrame()
for GW_event in events:
    print('Computing event {}'.format(GW_event))
    df = load_data_GWTC(GW_event)
    conv_df = convert_data(df)
    print(df.head())

        
    threads = 20
    indicies = np.arange(Nsamples)

    # with multiprocessing.Pool(threads) as p:
    #     SNRs = list(tqdm(p.imap(compute_SNR,indicies), total = Nsamples))
    # SNRs = np.array(SNRs).T
    pdet = pdet_rth(np.array(df.network_matched_filter_snr), ndet) ; ptheta = p_theta_omega(df, cosmo_use = False)
    # pdet = pdet_rth(np.array(SNRs), ndet) ; ptheta = p_theta_omega(df, cosmo_use = False)
    pdet = pdet.flatten() ; ptheta = ptheta.flatten()

    labelsfig = plt.figure(figsize=(15,10))

    Npoints = 500
    H0vec = np.linspace(20,140,Npoints)
    values= np.ones(Npoints)
    values_no_w = np.ones(Npoints)

    dH = np.diff(H0vec)[0]
    r = Nsamples
    likelihoods = [] 

    #print(conv_df)
    scaled_theta = scaler_x.transform(conv_df)

    for i in tqdm(range(r),desc = 'Calculating p(theta|Omega,D, I)'):
        # print(ptheta[i])
        like = p_theta_H0(scaled_theta[i,:],H0vec, hyper_dict)
        like  = like/np.sum(like*dH)
        # if np.isnan(like)[0] == 0:
        like_no_w = like
        like_with_w = like/(ptheta[i]*pdet[i])

        likelihoods.append(like)

        plt.plot(H0vec,like/np.sum(like*dH),alpha = 0.01, linewidth = 1)
        values += like_with_w#/np.sum(like_with_w*dH)
        values_no_w += like_no_w/np.sum(like_with_w*dH)
        
    values /= r
    post = values / np.sum( values*dH)
    post_no_w = values_no_w / np.sum( values_no_w*dH)

    short_names = ['GW150914', 'GW151226', 'GW170104', 'GW170608', 'GW170809', 'GW170814', 'GW170818', 'GW170823', 'GW190412', 'GW190425', 'GW190521','GW190814' ]

    for name in short_names:
        if GW_event == 'GW190521_074359':
            break 
        elif GW_event[:8] == name:
            event_name = name
            break
        else: 
            event_name = GW_event
            
    print(event_name)   
    O3_events_posteriors = json.load(open('../O3_Posteriors_file/O3_gwcosmo_H0_event_posteriors.json'))
    posterior_of_event = O3_events_posteriors['Mu_g_32.27_Mmax_112.5_band_K_Lambda_4.59'][event_name]
    H0_grid = O3_events_posteriors['H0_grid']        



    #Interpolate to normalize between 30-110 H0
    f = interpolate.interp1d(H0_grid, posterior_of_event)
    ynew = f(H0vec) 
    post_O3 = ynew/np.sum(ynew*dH)    
    
    
    
    ####EMPTY
    path_empty = '../../../o3-cosmology/gwcosmo_results/mature_circulation_material/results/Mu_g_32.27_Mmax_112.5_band_K_Lambda_4.59_empty/'
    if (run == 'O1') or (run == 'O2'):
        for name in short_names:
            if GW_event[:8] == name:
                empty_file = name
                break
            else: 
                empty_file = GW_event
    elif run == 'O3':
        data = os.listdir(path_empty) 
        event = GW_event[:8]
        for file in data:
            if file.startswith(event):
                empty_file = file
  
        
    path_file_npz = path_empty + empty_file+'/'+empty_file+'.npz' 
    data_empty = np.load(path_file_npz, allow_pickle=True)
    data_empty = data_empty['arr_0']
    H0_grid_empty = data_empty[0]
    posterior_empty = data_empty[2]

    JS = jensenshannon(post_O3, post)
    print('Jensen-Shannon value for {} = {}'.format(GW_event, JS))


    plt.plot(H0vec,post_O3, '--b', alpha=1, linewidth=5, label = 'O3 GWcosmo Posterior') 
    plt.plot(H0_grid_empty,posterior_empty, '--g', alpha=1, linewidth=5, label = 'O3 GWcosmo Posterior EMPTY') 

    plt.title(GW_event + '; JS = {}'.format(JS), fontsize = 20)
    plt.plot(H0vec,post, 'r', alpha=1, linewidth=5, label = '$p(H_{0} | \mathbf{h}, D)$, posterior')
    
    # plt.plot(H0vec,post_no_w, '--k', alpha=1, linewidth=5, label = '$p(H_{0} | \mathbf{h}, D)$, posterior_nocosmo')
    # plt.fill_between(H0vec,y_up, y_down, color = 'red', alpha = 0.5, label = '$1\sigma$')
    #plt.plot([], [],'k', alpha=0.1, label = '$p(\Theta | H_{0})$, likelihoods')

    plt.ylim([0.00,0.025])
    plt.xlim([20,140])
    plt.legend(loc = 'best', fontsize = 15)
    plt.grid(True, alpha = 0.5)

    plt.xlabel(r'$H_{0} \: [km \: s^{-1} \: Mpc^{-1}]$',fontsize = 25)
    plt.ylabel(r'$p(H_{0}) \: [km^{-1} \: s \: Mpc] $',fontsize = 25)
    plt.savefig(Folder+'/plots/'+GW_event)
    np.savetxt(Folder+'/posteriors/'+GW_event+'.txt',post)
#     np.savetxt(Folder+'/posteriors_no_w/'+GW_event+'no_w.txt',post_no_w)
    np.savetxt(Folder+'/O3_H0_post/O3_H0_'+GW_event+'.txt',post_O3)
    print('Saving Posterior')

    JS_dict = {'JS':JS, 'Mean_DL': np.mean(df.luminosity_distance), 'Mean_m1': np.mean(df.mass_1), 'Mean_m2':np.mean(df.mass_2), 'Mean_SNR': np.mean(df.network_matched_filter_snr)}
    
    df_JS = pd.DataFrame(JS_dict, index = [counter] )
    tot_JS = pd.concat([tot_JS, df_JS],  ignore_index=False)
    counter += 1
    
tot_JS.to_csv(Folder+'/JS_means/JS_values.csv')


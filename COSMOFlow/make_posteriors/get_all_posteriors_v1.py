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
from scipy import interpolate


import argparse


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

args = vars(ap.parse_args())
Folder = str(args['Name_folder'])
Nsamples= int(args['samples'])
rth= int(args['SNRth'])
Flow= str(args['Flow'])


#check if directory exists
if os.path.exists(Folder) is False:
    #Save model in folder
    os.mkdir(Folder)
    os.mkdir(Folder+'/plots')
    os.mkdir(Folder+'/posteriors')
    os.mkdir(Folder+'/posteriors_no_w')
    os.mkdir(Folder+'/O3_H0_post')

#'GW190425', 'GW190814','GW200115_042309','GW200105_162426' ??????
# 
# events = [ 'GW150914_095045','GW151226_033853', 'GW170104_101158', 'GW170608_020116', 
#          'GW170809_082821', 'GW170809_082821','GW170814_103043',  'GW170818_022509',    
#           'GW170823_131358',  'GW190408_181802', 'GW190412_053044', 'GW190503_185404', 
#          'GW190512_180714', 'GW190513_205428', 'GW190517_055101', 'GW190519_153544', 
#          'GW190521_030229', 'GW190521_074359', 'GW190602_175927', 'GW190630_185205', 
events = [
         'GW190701_203306', 'GW190706_222641', 'GW190707_093326', 'GW190708_232457', 
         'GW190720_000836', 'GW190727_060333', 'GW190728_064510', 
         'GW190828_063405', 'GW190828_065509', 'GW190910_112807', 'GW190915_235702'] 
#          'GW190924_021846', 'GW191109_010717', 'GW191129_134029', 'GW191204_171526', 
#          'GW191216_213338', 'GW191222_033537', 'GW200112_155838', 'GW200129_065458', 
#          'GW200202_154313', 'GW200224_222234', 'GW200225_060421', 'GW200311_115853']



def load_data_GWTC(event):
    if int(event[2:8]) <= 190930:
        
        path_gw = '/data/wiay/federico/PhD/GWTC_2.1/'
        file_name = path_gw+'IGWN-GWTC2p1-v2-{}_PEDataRelease_mixed_cosmo.h5'.format(event)
    else:   
        path_gw = '/data/wiay/federico/PhD/GWTC_3/'
        file_name = path_gw+'IGWN-GWTC3p0-v1-{}_PEDataRelease_mixed_nocosmo.h5'.format(event)
    
    d = h5py.File(file_name,'r')
    samples = np.array(d.get('C01:IMRPhenomXPHM/posterior_samples'))
    d.close()
    df = pd.DataFrame(samples)
    df = df[['luminosity_distance', 'mass_1', 'mass_2', 
           'a_1', 'a_2', 'tilt_1', 'tilt_2', 
           'ra', 'dec', 'theta_jn','phi_12',
           'psi','geocent_time','phi_jl' ]]
    df = df[(df['mass_1'] > 4.98) & (df['mass_2'] > 4.98)]
    df =  df.sample(frac=1).reset_index(drop=True)
    return df

#transform Polar into cartesian and spins to sigmoids
def convert_data(df):
    data = df
#transform Polar into cartesian and spins to sigmoids
    def spherical_to_cart(dl, ra, dec):

        x,y,z = spherical_to_cartesian(dl, dec, ra)
        return x,y,z

    coordinates= data[['luminosity_distance', 'ra', 'dec']]
    dl = np.array(coordinates.luminosity_distance)
    ra = np.array(coordinates.ra)
    dec = np.array(coordinates.dec)

    x,y,z = spherical_to_cart(dl, ra, dec)

    data['xcoord'] = x
    data['ycoord'] = y
    data['zcoord'] = z
    
    df[ 'geocent_time'] = abs(df.geocent_time % 86400)

    return data[['xcoord', 'ycoord', 'zcoord','mass_1', 'mass_2', 'theta_jn', 'psi', 'geocent_time']]


def logit_data(data_to_logit):
    a1_logit = logit(torch.from_numpy(np.array(data_to_logit.a1)))
    a2_logit = logit(torch.from_numpy(np.array(data_to_logit.a2)))
    phijl_logit = logit(torch.from_numpy(np.array(data_to_logit.phi_jl)))
    phi12_logit = logit(torch.from_numpy(np.array(data_to_logit.phi_12)))
    pol_logit = logit(torch.from_numpy(np.array(data_to_logit.polarization)))
    tc_logit = logit(torch.from_numpy(np.array(data_to_logit.geo_time)))

    data_to_logit.loc[:,'a1'] = np.array(a1_logit)
    data_to_logit.loc[:,'a2'] = np.array(a2_logit)
    data_to_logit.loc[:,'phi_jl'] = np.array(phijl_logit)
    data_to_logit.loc[:,'phi_12'] = np.array(phi12_logit)
    data_to_logit.loc[:,'polarization'] = np.array(pol_logit)
    data_to_logit.loc[:,'geo_time'] = np.array(tc_logit)
    return data_to_logit

def sigmoid_data(data_to_sigmoid):
    a1_sigmoid= sigmoid(torch.from_numpy(np.array(data_to_sigmoid.a1)))
    a2_sigmoid = sigmoid(torch.from_numpy(np.array(data_to_sigmoid.a2)))
    phijl_sigmoid = sigmoid(torch.from_numpy(np.array(data_to_sigmoid.phi_jl)))
    phi12_sigmoid = sigmoid(torch.from_numpy(np.array(data_to_sigmoid.phi_12)))
    pol_sigmoid = sigmoid(torch.from_numpy(np.array(data_to_sigmoid.polarization)))
    tc_sigmoid = sigmoid(torch.from_numpy(np.array(data_to_sigmoid.geo_time)))

    data_to_sigmoid.loc[:,'a1'] = np.array(a1_sigmoid)
    data_to_sigmoid.loc[:,'a2'] = np.array(a2_sigmoid)
    data_to_sigmoid.loc[:,'phi_jl'] = np.array(phijl_sigmoid)
    data_to_sigmoid.loc[:,'phi_12'] = np.array(phi12_sigmoid)
    data_to_sigmoid.loc[:,'polarization'] = np.array(pol_sigmoid)
    data_to_sigmoid.loc[:,'geo_time'] = np.array(tc_sigmoid)
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



for GW_event in events:
    print('Computing event {}'.format(GW_event))
    df = load_data_GWTC(GW_event)
    conv_df = convert_data(df)
    print(conv_df.head())
 
    threads = 10

    def compute_pDet_PTheta(i):
        pdet = [] 
        ptheta = []
        r_th = rth

        theta = [df.luminosity_distance[i], df.mass_1[i]
                 , df.mass_2[i], df.a_1[i], df.a_2[i], df.tilt_1[i], df.tilt_2[i], df.ra[i], df.dec[i], df.theta_jn[i],
                 df.phi_jl[i], df.phi_12[i],df.psi[i], df.geocent_time[i]%86400]

        pdet.append(pdet_theta.p_D_theta(theta, r_th))
        ptheta.append(pdet_theta.p_theta_omega(theta))

        return np.array(pdet), np.array(ptheta)


    with multiprocessing.Pool(threads) as p:
        pdet_ptheta = list(tqdm(p.imap(compute_pDet_PTheta,np.arange(Nsamples)), total = Nsamples, desc = 'Calculating p(D|theta) and p(theta|Omega_0)'))    
    pdet_ptheta = np.array(pdet_ptheta)

    pdet = pdet_ptheta[:, 0] ; ptheta = pdet_ptheta[:, 1] 

    #Get flow 
    flow, hyper_dict, scaler_x, scaler_y = load_hyperparameters_scalers_flow(Flow)   


    def p_theta_H0(theta, conditional):
        dCon = np.diff(conditional)[0]
        Y_H0_conditional = scaler_y.transform(conditional.reshape(-1,1))
        samples = scaler_x.transform(np.array(theta).reshape(1,-1))

        dict_rand = {'x':list(samples[:,0]), 'y':list(samples[:,1]), 'z':list(samples[:,2]), 
                      'm1':list(samples[:,3]), 'm2':list(samples[:,4]),'a1':list(samples[:,5]),
                       'a2':list(samples[:,6]), 'tilt1':list(samples[:,7]), 'tilt2':list(samples[:,8]),
                      'theta_jn':list(samples[:,9]), 'phi_jl':list(samples[:,10]), 'phi_12':list(samples[:,11]),
                      'polarization':list(samples[:,12]), 'geo_time':list(samples[:,13])}

        samples = pd.DataFrame(dict_rand)
        scaled_theta = samples 
        if hyper_dict['log_it'] == 1:
            logit_data(scaled_theta)
            scaled_theta = scaled_theta[np.isfinite(scaled_theta).all(1)]

        scaled_theta = np.array(scaled_theta)

        scaled_theta = scaled_theta.T*np.ones((1,len(Y_H0_conditional)))



        conditional = np.array(Y_H0_conditional)
        data = np.array(conditional)
        data_scaled = torch.from_numpy(data.astype('float32'))

        def Flow_posterior(target, conditional): 
            flow.eval()
            flow.to('cpu')

            with torch.no_grad():

                logprobx = flow.log_prob(target, conditional=conditional.to('cpu'))
                logprobx = logprobx.numpy() 



                return  logprobx

        Log_Prob = Flow_posterior(torch.from_numpy(scaled_theta.T).float(), data_scaled)


        return np.exp(Log_Prob) /np.sum(np.exp(Log_Prob)*dCon) 




    labelsfig = plt.figure(figsize=(15,10))

    Npoints = 500
    H0vec = np.linspace(20,140,Npoints)
    values= np.ones(Npoints)
    values_no_w = np.ones(Npoints)

    dH = np.diff(H0vec)[0]
    r = Nsamples
    likelihoods = [] 

    #print(conv_df)


    for i in tqdm(range(r),desc = 'Calculating p(theta|Omega,D, I)'):

        like = p_theta_H0(conv_df.iloc[i,:],H0vec)
        if np.isnan(like)[0] == 0:
            like_no_w = like
            like_with_w = like/(ptheta[i]*pdet[i])

            likelihoods.append(like_with_w/np.sum(like_with_w*dH))

            plt.plot(H0vec,like_with_w/np.sum(like_with_w*dH), 'k', alpha=0.05, linewidth = 1)
            values += like_with_w
            values_no_w += like_no_w

    post = values / np.sum( values*dH)
    post_no_w = values_no_w / np.sum( values_no_w*dH)

    short_names = ['GW150914', 'GW151226', 'GW170104', 'GW170608', 'GW170809', 'GW170814', 'GW170818', 'GW170823', 'GW190412', 'GW190425', 'GW190521','GW190814' ]
    
    for name in short_names:
        if GW_event[:8] == name:
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


    plt.plot(H0vec,post_O3, 'b', alpha=1, linewidth=5, label = 'O3 GWcosmo Posterior')    

    plt.title('GW'+GW_event, fontsize = 20)
    plt.plot(H0vec,post, 'k', alpha=1, linewidth=5, label = '$p(H_{0} | \mathbf{h}, D)$, posterior')
    plt.plot(H0vec,post_no_w, 'r', alpha=1, linewidth=5, label = '$p(H_{0} | \mathbf{h}, D)$, posterior_unweighted')
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
    np.savetxt(Folder+'/posteriors_no_w/'+GW_event+'no_w.txt',post_no_w)
    np.savetxt(Folder+'/O3_H0_post/O3_H0_'+GW_event+'.txt',post_O3)
    print('Saving Posterior')


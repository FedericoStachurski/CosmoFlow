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


import argparse


#pass arguments 
# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-Folder", "--Name_folder", required=True,
   help="Name of the folder to save the GW_posteriors")
ap.add_argument("-GW", "--GW_event", required=True,
   help="GW_event_to use")
ap.add_argument("-Nsamples", "--samples", required=True,
   help="Posterior samples to use")
ap.add_argument("-SNRth", "--SNRth", required=True,
   help="SNR threshold")
ap.add_argument("-Flow", "--Flow", required=True,
   help="Trained flow to use")

args = vars(ap.parse_args())
Folder = str(args['Name_folder'])
GW_event = str(args['GW_event'])
Nsamples= int(args['samples'])
rth= int(args['SNRth'])
Flow= str(args['Flow'])


#check if directory exists
if os.path.exists(Folder) is False:
    #Save model in folder
    os.mkdir(Folder)



path_gw21 = '/data/wiay/federico/PhD/GWTC_2.1/'
# event = ['150914_095045','151012_095443','151226_033853','170104_101158','170608_020116',
#          '170729_185629', '170809_082821', '170814_103043','170818_022509','170823_131358']



file_name = path_gw21+'IGWN-GWTC2p1-v2-GW{}_PEDataRelease_mixed_cosmo.h5'.format(GW_event)

d = h5py.File(file_name,'r')
samples = np.array(d.get('C01:IMRPhenomXPHM/posterior_samples'))
d.close()

df = pd.DataFrame(samples)


df = df[[ 'luminosity_distance', 'mass_1', 'mass_2', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'ra', 'dec', 'theta_jn']]
 

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

    spins = data[['a_1','a_2']]

    a1_logit = logit(torch.from_numpy(np.array(spins.a_1)))
    a2_logit = logit(torch.from_numpy(np.array(spins.a_2)))

    data['a1_logit'] = a1_logit
    data['a2_logit'] = a2_logit



    return data[['xcoord', 'ycoord', 'zcoord','mass_1', 'mass_2','a1_logit', 'a2_logit', 'tilt_1', 'tilt_2', 'theta_jn']]


conv_df = convert_data(df)
df = pd.DataFrame(samples)
df = df[[ 'luminosity_distance', 'mass_1', 'mass_2', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'ra', 'dec', 'theta_jn']]



rth = rth

print('Calculating p(D|theta) and p(theta|Omega_0)')
pdet = [] 
ptheta = []
for i in tqdm(range(Nsamples)):
    theta = [df.luminosity_distance[i], df.mass_1[i]
             , df.mass_2[i], df.a_1[i], df.a_2[i], df.tilt_1[i], df.tilt_2[i], df.ra[i], df.dec[i], df.theta_jn[i], 0, 0, 0, 0]

    pdet.append(pdet_theta.p_D_theta(theta, rth))
    ptheta.append(pdet_theta.p_theta_omega(theta))
    
    
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
            linear_transform=None)
    elif flow_type == 'CouplingNSF':   
            flow_empty = CouplingNSF(n_inputs= n_inputs,
            n_transforms= n_transforms,
            n_neurons= n_neurons,
            n_conditional_inputs = n_conditional_inputs,
            n_blocks_per_transform = n_blocks_per_transform,
            batch_norm_between_transforms=True,
            dropout_probability=dropout,
            linear_transform=None)
    
    flow_empty.load_state_dict(flow_load)
    flow = flow_empty
    flow.eval()
    
    return flow, hyperparameters, scaler_x, scaler_y




flow, hyper_dict, scaler_x, scaler_y = load_hyperparameters_scalers_flow(Flow)   


def p_theta_H0(theta, conditional):
    torch.manual_seed(np.random.randint(10000))
    dCon = np.diff(conditional)[0]
    Y_H0_conditional = scaler_y.transform(conditional.reshape(-1,1))
    scaled_theta = scaler_x.transform(np.array(theta).reshape(1,-1))
    
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
H0vec = np.linspace(30,110,Npoints)
values= np.ones(Npoints)
values_no_w = np.ones(Npoints)
 
dH = np.diff(H0vec)[0]
r = Nsamples
likelihoods = [] 
for i in tqdm(range(r)):
    
    like = p_theta_H0(conv_df.iloc[i,:],H0vec)
    #like_no_w = p_theta_H0(df.iloc[i,:],H0vec)
    if np.isnan(like)[0] == 0:  
        like /= (ptheta[i]*pdet[i])

        likelihoods.append(like/np.sum(like*dH))

        #plt.plot(H0vec,like/np.sum(like*dH), 'k', alpha=0.1, linewidth = 1)
        values += like/np.sum(like*dH)
#         values_no_w += like_no_w

        post = values / np.sum( values*dH)
        post_no_w = values_no_w / np.sum( values_no_w*dH)
#     plt.plot(H0vec,post, 'r', alpha=(i+1)/(2*r))


errors = []
for i in range(Npoints):
    errors.append(np.sqrt(np.var(np.vstack(np.array(likelihoods))[:,i])))
    


#post = np.exp(values) / np.sum( np.exp(values)*dH)@
post = values / np.sum( values*dH)

y_up = post + np.array(errors)
y_down = post - np.array(errors)

#Open O3 posterior
path_O3 = '/data/wiay/federico/PhD/O3_posteriors/'
 
event_O3 = path_O3 +'GW'+GW_event[0:6]
import os
from scipy import interpolate
if os.path.isdir(event_O3):

    with np.load(event_O3+'/'+'GW'+GW_event[0:6]+'.npz', allow_pickle=True) as data:
        data = data['arr_0']
        
    #Interpolate to normalize between 30-110 H0
    f = interpolate.interp1d(data[0], data[2])
    ynew = f(H0vec) 
    post_O3 = ynew/np.sum(ynew*dH)
        
        
    plt.plot(H0vec,post_O3, 'b', alpha=1, linewidth=5, label = 'O3 GWcosmo Posterior')    
        






plt.title('GW'+GW_event, fontsize = 20)
plt.plot(H0vec,post, '--k', alpha=1, linewidth=5, label = '$p(H_{0} | \mathbf{h}, D)$, posterior')
plt.fill_between(H0vec,y_up, y_down, color = 'red', alpha = 0.5, label = '$1\sigma$')
#plt.plot([], [],'k', alpha=0.1, label = '$p(\Theta | H_{0})$, likelihoods')

plt.ylim([0.00,0.025])
plt.xlim([30,110])
plt.legend(loc = 'best', fontsize = 25)
plt.grid(True, alpha = 0.5)

plt.xlabel(r'$H_{0} \: [km \: s^{-1} \: Mpc^{-1}]$',fontsize = 25)
plt.ylabel(r'$p(H_{0}) \: [km^{-1} \: s \: Mpc] $',fontsize = 25)
plt.savefig(Folder+'/'+GW_event)
np.savetxt(Folder+'/'+GW_event+'.txt',post)


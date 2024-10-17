import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import h5py
import numpy as np 
import pandas as pd 
from cosmology_functions import utilities 
from scipy.stats import truncnorm
from train_flow.handle_flow import Handle_Flow
import argparse
from tqdm import tqdm 
import h5py
import time 
import pickle 
import matplotlib.pyplot as plt
from gw_functions.gw_event_names import GW_events
# GW_class = GW_events()

ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-runs", "--runs", nargs='+', required=True,
   help="make data from detector: OPTIONS [O1, O2, O3..]", default = 'O3')
ap.add_argument("-flow_name", "--flow_name", required=True,
   help="name_of_flow")
ap.add_argument("-Ntheta", "--Ntheta", required=True,
   help="Number of GW posterior points to be used in batching the flow", default = 100)
ap.add_argument("-Nomega", "--Nomega", required=True,
   help="Number of conditional points to be used", default = 100)
ap.add_argument("-Ntotal", "--Ntotal", required=True,
   help="Total points to sample", default = 100)
ap.add_argument("-device", "--device", required=True,
   help="cpu, cuda:0 ...", default = 'cuda:0')



args = vars(ap.parse_args())
runs = str(args['runs'])
flow_name = str(args['flow_name'])
N_theta = int(args['Ntheta'])
N_samples= int(args['Nomega'])
N_total = int(args['Ntotal'])
device = str(args['device'])




os.chdir("..")
path = 'train_flow/trained_flows_and_curves/'

def get_data_GW(events, N_theta):
    data_total = pd.DataFrame()
    for event in tqdm(events): 
        df = utilities.load_data_GWTC(event)
        df = df[['luminosity_distance', 'mass_1', 'mass_2', 'ra', 'dec']].head(N_theta)
        data = [data_total, df]
        data_total = pd.concat(data)
    return data_total


if 'O1' in runs:
    flow_name_o1 = flow_name+'_O1'
    flow_class_o1 = Handle_Flow(path, flow_name_o1, device, epoch = None)
    data_o1 = get_data_GW(GW_class.get_event('O1', 'HL'), N_theta)
    n_conditional = flow_class_o1.hyperparameters['n_conditional_inputs']
    
if 'O2' in runs:
    flow_name_o2_hlv = flow_name+'_O2_HLV'
    flow_class_o2_hlv = Handle_Flow(path, flow_name_o2_hlv, device, epoch = None)
    data_o2_hlv = get_data_GW(GW_class.get_event('O2', 'HLV'), N_theta)
    n_conditional = flow_class_o2_hlv.hyperparameters['n_conditional_inputs'] 

    flow_name_o2_hl = flow_name+'_O2_HL'
    flow_class_o2_hl = Handle_Flow(path, flow_name_o2_hl, device, epoch = None)
    data_o2_hl = get_data_GW(GW_class.get_event('O2', 'HL'), N_theta)
    
    
if 'O3' in runs: 
    flow_name_o3_hlv = flow_name
    flow_class_o3_hlv = Handle_Flow(path, flow_name_o3_hlv, device, epoch = None)
    n_conditional = flow_class_o3_hlv.hyperparameters['n_conditional_inputs']
    gw_event_name_parameters = {'detectors':'HLV', 'run':'O3', 'population': 'BBH'}
    gw_event_name_class = GW_events(gw_event_name_parameters)
    hlv_events = gw_event_name_class.get_event()
    hlv_events = hlv_events[:2]
    # hlv_events = GW_class.get_event('O3', 'HLV') 
    # hl_events = GW_class.get_event('O3', 'HL') 
    # lv_events = GW_class.get_event('O3', 'LV')
    # hv_events = GW_class.get_event('O3', 'HV')
    
    N_hlv = len(hlv_events)
    # N_hl = len(hl_events)
    # N_lv = len(lv_events)
    # N_hv = len(hv_events)
    
    # N_tot_events_o3 = N_hl + N_lv + N_hv + N_hlv
    N_tot_events_o3 = N_hlv
    
    data_o3_hlv = get_data_GW(hlv_events, N_theta)
    # data_o3_hl = get_data_GW(hl_events, N_theta)
    # data_o3_lv = get_data_GW(lv_events, N_theta)
    # data_o3_hv = get_data_GW(hv_events, N_theta)
    data_o3_hlv = pd.merge(data_o3_hlv, utilities.make_det_setup_dataframe(['H1','L1', 'V1'], N_theta), left_index=True, right_index=True)
    # data_o3_hl = pd.merge(data_o3_hl, utilities.make_det_setup_dataframe(['H1','L1'], N_theta), left_index=True, right_index=True)
    # data_o3_lv = pd.merge(data_o3_lv, utilities.make_det_setup_dataframe(['L1','V1'], N_theta), left_index=True, right_index=True)
    # data_o3_hv = pd.merge(data_o3_hv, utilities.make_det_setup_dataframe(['H1','V1'], N_theta), left_index=True, right_index=True)
    # data_tot_o3 = pd.concat((data_o3_hlv, data_o3_hl, data_o3_lv, data_o3_hv))
    data_tot_o3 = data_o3_hlv
    


def posterior_eval_flow(data, prior_samples, flow_class_variable, N_events, N_theta, N_samples):
    likelihood_eval = flow_class_variable.p_theta_H0_one_go_batch(data, prior_samples.T)
    likelihood_eval = np.reshape(likelihood_eval, (int(N_events*N_theta),int(N_samples)))
    posterior_eval = flow_class_variable.get_posterior_from_batch(likelihood_eval, N_samples,  N_theta , N_events)
    return posterior_eval


########## Start rejection sampling
accepted_points = []
ESS_list, N_accept_list = [], []
wi_list = []
samples_proposal = []
time_list = [] 
counter = 0 
probs = [] 


while True:
    start = time.time()
    prior_samples = utilities.make_samples(N_samples, dimensions = 2)
    log_prob = np.zeros(N_samples)
    # likelihood_eval = flow_class.p_theta_H0_one_go_batch(data_total, prior_samples.T)
    # probs.append(likelihood_eval)
    # likelihood_eval = np.reshape(likelihood_eval, (int(Nevents*Ntheta),int(Nomega)))
    # posterior_eval = flow_class.get_posterior_from_batch(likelihood_eval, Nomega,  Ntheta , Nevents)
    if 'O1' in runs:
        posterior_o1 = posterior_eval_flow(data_o1, prior_samples, flow_class_o1, len(GW_class.get_event('O1', 'HL')), N_theta, N_samples)
        log_prob += np.log(posterior_o1)
    
    if 'O2' in runs:
        posterior_o2_hlv = posterior_eval_flow(data_o2_hlv, prior_samples, flow_class_o2_hlv, len(GW_class.get_event('O2', 'HLV')), N_theta, N_samples)
        posterior_o2_hl = posterior_eval_flow(data_o2_hl, prior_samples, flow_class_o2_hl, len(GW_class.get_event('O2', 'HL')), N_theta, N_samples)
        log_prob += np.log(posterior_o2_hlv) + np.log(posterior_o2_hlv)
        
    if 'O3' in runs:    
        # posterior_o3_hlv = posterior_eval_flow(data_o3_hlv, prior_samples, flow_class_o3_hlv, len(GW_class.get_event('O3', 'HLV')), N_theta, N_samples)
        posterior_o3 = posterior_eval_flow(data_tot_o3, prior_samples, flow_class_o3_hlv, N_tot_events_o3, N_theta, N_samples)
        # posterior_o3_lv = posterior_eval_flow(data_o3_lv, prior_samples, flow_class_o3_hlv, len(GW_class.get_event('O3', 'LV')), N_theta, N_samples)
        # posterior_o3_hv = posterior_eval_flow(data_o3_hv, prior_samples, flow_class_o3_hlv, len(GW_class.get_event('O3', 'HV')), N_theta, N_samples)
        log_prob += posterior_o3
        
    log_w_i_unweighted = log_prob
    # print(log_w_i_unweighted)
    wi_list.append(log_w_i_unweighted)
    
    N = np.sum(np.exp(np.array(wi_list) - np.max(wi_list)))
    # print(np.exp(np.array(wi_list) - np.max(wi_list)))
    # ESS = np.sum(2*np.array(wi_list))**2 / (np.sum(np.array(wi_list)**2))
    samples_proposal.append(prior_samples)
    end = time.time()
    time_list.append(end - start)
    counter += 1
    efficiency = counter * N_samples / N
    sys.stdout.write('\r | N = {} | iteration = {} | time = {} | total_cumulative_time = {} | efficiency = {} '.format(round(N,2),counter,round(end - start,2), round(np.sum(np.array(time_list)),2),round(efficiency),5))
    
    N_accept_list.append(N)
    plt.plot(np.cumsum(np.array(time_list)), np.array(N_accept_list), '-k', linewidth = 4)
    plt.xlabel(r'time [$s$]', fontsize = 13)
    plt.ylabel(r'$N = <w_{i}> $', fontsize = 13)
    plt.axhline(y = N_total ,color = 'r', linewidth = 3,  label = 'Ntotal')
    plt.grid(True)
    plt.yscale('log')
    plt.savefig('make_posteriors/posterior_samples_rejection/statistics_rejection_sampling.png', dpi = 300)
    plt.show()
    plt.close()
    
    
    if N >= N_total:
        break

times = time_list
weights = np.concatenate(wi_list)
weights = np.exp(weights - np.max(weights))
samples_from_prior = np.concatenate(samples_proposal, axis = 1)
N_points = len(weights)
# weights -= np.max(weights)
t_i = np.random.uniform(0,1, N_points)
inx_accept = np.where(t_i < weights)[0]
accepted_points = samples_from_prior[:,inx_accept]

# dictionary_samples = {'H0':accepted_points[0,:], 'gamma':accepted_points[1,:], 'kappa':accepted_points[2,:], 'zp':accepted_points[3,:],
#                      'alpha':accepted_points[4,:], 'beta':accepted_points[5,:], 'mmax':accepted_points[6,:], 'mmin':accepted_points[7,:],
#                      'mu_g':accepted_points[8,:], 'sigma_g':accepted_points[9,:], 'lambda_peak':accepted_points[10,:], 'delta_m':accepted_points[11,:], 'weights':weights, 'times':times}


# dictionary_samples = {'H0':accepted_points[0,:], 'gamma':accepted_points[1,:],
#                       'mmax':accepted_points[2,:], 'mu_g':accepted_points[3,:],  'weights':weights, 'times':times}


dictionary_samples = {'H0':accepted_points[0,:], 'Om0':accepted_points[1,:],  'weights':weights, 'times':times}

# dictionary_data = { }

# posterior_samples = pd.DataFrame(dictionary_samples)
# posterior_data = pd.DataFrame(dictionary_data)
# posterior_samples.to_csv('make_posteriors/posterior_samples_rejection/run_{}_FLOW_{}_samples.csv'.format(run,flow_name))
with open('make_posteriors/posterior_samples_rejection/_FLOW_{}_Ntheta_{}_Nomega_{}_Ntotal_{}_data.pickle'.format(flow_name, N_theta, N_samples, N_total), 'wb') as handle:
        pickle.dump(dictionary_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
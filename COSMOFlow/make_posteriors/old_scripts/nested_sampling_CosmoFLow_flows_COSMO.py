import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import h5py
import numpy as np 
import pandas as pd 
from cosmology_functions import utilities 
from scipy.stats import truncnorm
# from train_flow.handle_flow import Handle_Flow
from train_flow.handle_flow_Chris import Handle_Flow
import argparse
from tqdm import tqdm 
import h5py
import time 
import pickle 
import matplotlib.pyplot as plt
from gw_functions.gw_event_names import GW_events
from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger
from nessai.utils.multiprocessing import initialise_pool_variables
parameters = {'population':'BBH'}
GW_class = GW_events(parameters)

ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-runs", "--runs", nargs='+', required=True,
   help="make data from detector: OPTIONS [O1, O2, O3..]", default = 'O3')
ap.add_argument("-flow_name", "--flow_name", required=True,
   help="name_of_flow")
ap.add_argument("-folder_name", "--folder_name", required=True,
   help="name_of_folder")
ap.add_argument("-Ntheta", "--Ntheta", required=True,
   help="Number of GW posterior points to be used in batching the flow", default = 100)
ap.add_argument("-Nomega", "--Nomega", required=True,
   help="Number of conditional points to be used", default = 100)
ap.add_argument("-Ntotal", "--Ntotal", required=True,
   help="Total points to sample", default = 100)
ap.add_argument("-device", "--device", required=True,
   help="cpu, cuda:0 ...", default = 'cuda:0')
ap.add_argument("-sampler", "--sampler", required=True,
   help="Rejection or NESSAI (nested)", default = 'Rejection')
ap.add_argument("-live_points", "--live_points", required=False,
   help="Live points to use", default = 1000)
ap.add_argument("-data_type", "--data_type", required=True,
   help="Real or Testing", default = 'Testing')




args = vars(ap.parse_args())
runs = str(args['runs'])
flow_name = str(args['flow_name']) #H0gammakzp
folder_name = str(args['folder_name']) #H0gammakzp
N_theta = int(args['Ntheta'])
N_samples= int(args['Nomega'])
N_total = int(args['Ntotal'])
device = str(args['device'])
sampler = str(args['sampler'])
data_type = str(args['data_type'])
live_points = int(args['live_points'])

# output = "posterior_samples_rejection/nested_sampling_folders/{}_run_{}_FLOW_{}_data".format(name,runs[0],flow_name)
# cosmoflow/COSMOFlow/data_cosmoflow/galaxy_catalog/testing_data_from_MLP/run_O3_det_['H1', 'L1', 'V1']_name_BBH_O3_events_all_para_catalog_True_band_K_batch_1_N_250_SNR_11_Nelect_4__Full_para_v1.csv
# print(os.getcwd())

def read_data(batch):
    path_name ="data_cosmoflow/galaxy_catalog/testing_data_from_MLP/"
    data_name = "run_O3_det_['H1', 'L1', 'V1']_name_BBH_O3_NEW_multipara_catalog_True_band_K_batch_1_N_250_SNR_11_Nelect_10__Full_para_v1.csv".format(batch)
    GW_data = pd.read_csv(path_name+data_name,skipinitialspace=True)
    print('Showing some of the data')
    print(GW_data.head(250))
    return GW_data



os.chdir("..")
path = 'train_flow/trained_flows_and_curves/'

def get_data_GW(events, N_theta):
    i = 0 
    data_total = pd.DataFrame()
    for event in tqdm(events): 
        df = utilities.load_data_GWTC(event)
        # df = df[['luminosity_distance', 'ra', 'dec', 'mass_1', 'mass_2']].head(N_theta)
        df = df[['luminosity_distance', 'ra', 'dec', 'mass_1', 'mass_2',
                 'a_1', 'a_2','tilt_1', 'tilt_2', 'theta_jn', 'phi_jl',
                 'phi_12', 'psi','geocent_time']].head(N_theta)
        df['geocent_time'] = utilities.convert_gps_sday(df['geocent_time'])
        data = [data_total, df]
        data_total = pd.concat(data)
        i+=1
    return data_total


if 'O1' in runs:
    
    flow_name_o1 = flow_name+'_O1_H1_L1_REALnvp_v1'
    flow_class_o1 = Handle_Flow(path, flow_name_o1, device, epoch = None)
    data_o1 = get_data_GW(GW_class.get_event('O1', 'HL'), N_theta)
    n_conditional = flow_class_o1.hyperparameters['n_conditional_inputs']
    
if 'O2' in runs:
    flow_name_o2_hlv = flow_name+'_O2_H1_L1_V1_REALnvp_v1'
    flow_class_o2_hlv = Handle_Flow(path, flow_name_o2_hlv, device, epoch = None)
    data_o2_hlv = get_data_GW(GW_class.get_event('O2', 'HLV'), N_theta)
    n_conditional = flow_class_o2_hlv.hyperparameters['n_conditional_inputs'] 

    flow_name_o2_hl = flow_name+'_O2_H1_L1_REALnvp_v1'
    flow_class_o2_hl = Handle_Flow(path, flow_name_o2_hl, device, epoch = None)
    data_o2_hl = get_data_GW(GW_class.get_event('O2', 'HL'), N_theta)
    
    
if 'O3' in runs: 
    flow_name_o3_hlv = flow_name+'_O3_H1_L1_14target_4cond_1_COSMO'
    flow_class_o3_hlv = Handle_Flow(path, flow_name_o3_hlv, device, epoch = None)
    if data_type == 'Real':
        N_evetns_o3_hlv = len(GW_class.get_event('O3', 'HLV'))
        data_o3_hlv = get_data_GW(GW_class.get_event('O3', 'HLV'), N_theta)
    
    elif data_type == 'Testing':
        print(os.getcwd())
        list_data = [] 
        for i in range(1):
            list_data.append(read_data(i+1))
            
        dataGW = pd.concat(list_data)
        dataGW = dataGW.drop_duplicates(keep='first').sample(frac=1)  
        data_o3_hlv =dataGW[['luminosity_distance', 'ra', 'dec', 'mass_1', 'mass_2',
                 'a_1', 'a_2','tilt_1', 'tilt_2', 'theta_jn', 'phi_jl',
                 'phi_12', 'psi','geocent_time']]
        data_o3_hlv['geocent_time'] = utilities.convert_gps_sday(data_o3_hlv['geocent_time'])
        N_evetns_o3_hlv = len(data_o3_hlv)
    
    n_conditional = flow_class_o3_hlv.hyperparameters['n_conditional_inputs']
    
    
    # flow_name_o3_hl = flow_name+'_O3_H1_L1_SPLINE_v1'
    # flow_class_o3_hl = Handle_Flow(path, flow_name_o3_hl, device, epoch = None)
    # data_o3_hl = get_data_GW(GW_class.get_event('O3', 'HL'), N_theta)
    
    # flow_name_o3_hv = flow_name+'_O3_H1_L1_SPLINE_v1'
    # flow_class_o3_hv = Handle_Flow(path, flow_name_o3_hv, device, epoch = None)
    # data_o3_hv = get_data_GW(GW_class.get_event('O3', 'HV'), N_theta)
    
    # flow_name_o3_lv = flow_name+'_O3_H1_L1_SPLINE_v1'
    # flow_class_o3_lv = Handle_Flow(path, flow_name_o3_hl, device, epoch = None)
    # data_o3_lv = get_data_GW(GW_class.get_event('O3', 'LV'), N_theta)
    


if sampler == 'Rejection':
    # ########## Start rejection sampling
    
    wi_list = []
    
    # probs = [] 
    log_posterior = []
    N_vals = []
    ESS_vals = []
    N_accept_list = [] 
    samples_proposal = []
    time_list = [] 
    counter = 0 
    
    while True:
        start = time.time()
        # 'H0', 'gamma', 'k', 'zp', 'beta', 'alpha', 'mmax', 'mmin', 'mu_g', 'sigma_g', 'lambda_peak','delta_m'
        prior_parameters = {'H0':[20,180], 'Om0':[0,1], 'w0':[-1.9,0.9], 'eta':[0.5,3]}
        prior_samples =utilities.prior_samples(N_samples,prior_parameters)
        log_prob = np.zeros(N_samples)
    
        if 'O1' in runs:
            N_events_o1 = len(GW_class.get_event('O1', 'HL'))
            log_posterior_o1 = flow_class_o1.temp_funct_post(flow_class_o1, data_o1, prior_samples, N_events_o1, N_theta, N_samples, ndim_target = 5)
            log_prob += log_posterior_o1 
        
        if 'O2' in runs:
            N_evetns_o2_hlv = len(GW_class.get_event('O2', 'HLV'))
            N_evetns_o2_hl = len(GW_class.get_event('O2', 'HL'))
            log_posterior_o2_hlv = flow_class_o2_hlv.temp_funct_post(flow_class_o2_hlv, data_o2_hlv, prior_samples, N_evetns_o2_hlv, N_theta, N_samples, ndim_target = 5)
            log_posterior_o2_hl = flow_class_o2_hl.temp_funct_post(flow_class_o2_hl, data_o2_hl, prior_samples, N_evetns_o2_hl, N_theta, N_samples, ndim_target = 5)
            
            log_prob += log_posterior_o2_hlv + log_posterior_o2_hl 
            
            
        if 'O3' in runs:    

            N_evetns_o3_hlv = 3#len(GW_class.get_event('O3', 'HLV'))
            # N_evetns_o3_hl = len(GW_class.get_event('O3', 'HL'))
            # N_evetns_o3_lv = len(GW_class.get_event('O3', 'LV'))
            # N_evetns_o3_hv = len(GW_class.get_event('O3', 'HV'))
            
            log_posterior_o3_hlv = flow_class_o3_hlv.temp_funct_post(flow_class_o3_hlv, data_o3_hlv, prior_samples, N_evetns_o3_hlv, N_theta, N_samples, ndim_target = 14)
            # log_posterior_o3_hl = flow_class_o3_hl.temp_funct_post(flow_class_o3_hl, data_o3_hl, prior_samples, N_evetns_o3_hl, N_theta, N_samples, ndim_target = 5)
            # log_posterior_o3_lv = flow_class_o3_lv.temp_funct_post(flow_class_o3_lv, data_o3_lv, prior_samples, N_evetns_o3_lv, N_theta, N_samples, ndim_target = 5)
            # log_posterior_o3_hv = flow_class_o3_hv.temp_funct_post(flow_class_o3_hv, data_o3_hv, prior_samples, N_evetns_o3_hv, N_theta, N_samples, ndim_target = 5)

            log_prob += log_posterior_o3_hlv 
            # log_prob += log_posterior_o3_hl
            # log_prob += log_posterior_o3_lv
            # log_prob += log_posterior_o3_hv
            
        
        #### Compute statistics and weights
        log_posterior.append(log_prob)
        wj = np.concatenate(log_posterior)
        wi_list.append(wj)
        
        wj = np.exp(wj - np.max(wj))
        
        N = np.sum(wj)
        ESS = np.sum(wj)**(2)  /  np.sum(wj**(2))
        
        N_vals.append(N)
        ESS_vals.append(ESS)
        
        counter += 1
        efficiency = counter * N_samples / N
        
        samples_proposal.append(prior_samples)
        end = time.time()
        time_list.append(end - start)
        sys.stdout.write('\rESS = {} | N = {} | iteration = {} | time = {} | total_cumulative_time = {} | efficiency = {} '.format(round(ESS,2), round(N,2),counter,round(end - start,2), round(np.sum(np.array(time_list)),2),round(efficiency),2))
        
        N_accept_list.append(N)
        plt.plot(np.cumsum(np.array(time_list)), np.array(N_accept_list), '-k', linewidth = 4)
        plt.xlabel(r'time [$s$]', fontsize = 13)
        plt.ylabel(r'$N = <w_{i}> $', fontsize = 13)
        plt.axhline(y = N_total ,color = 'r', linewidth = 3,  label = 'Ntotal')
        plt.grid(True)
        plt.yscale('log')
        plt.savefig('make_posteriors/posterior_samples_rejection/statistics_rejection_sampling_H0gammakzp.png', dpi = 300)
        plt.show()
        plt.close()
        
        
        if N >= N_total:
            break
    

    
    prior_points = np.concatenate(samples_proposal, axis = 1)
    logs_temp = np.concatenate(log_posterior)
    weights_temp = logs_temp - np.max(logs_temp)
    t_i = np.log(np.random.uniform(0,1, len(logs_temp)))
    inx_accept = np.where(t_i < weights_temp)[0]
    accepted_points = prior_points[:,inx_accept]
    
    # dictionary_samples = {'H0':accepted_points[0,:], 'gamma':accepted_points[1,:], 'k':accepted_points[2,:], 'zp':accepted_points[3,:]}
    dictionary_samples = {'H0':accepted_points[0,:], 'Om0':accepted_points[1,:],'w0':accepted_points[3,:], 'eta':accepted_points[2,:]}
    # 'weights':weights, 'times':times}
    
    with open('make_posteriors/posterior_samples_rejection/FLOW_{}_Ntheta_{}_Nomega_{}_Ntotal_{}_data.pickle'.format(flow_name,N_theta, N_samples, N_total), 'wb') as handle:
            pickle.dump(dictionary_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)





elif sampler == 'Nested_AI':
    os.chdir("make_posteriors/posterior_samples_rejection/")
    output = "nested_sampling_folders/NESSAI_{}_run_{}_FLOW_{}_TEST".format(folder_name,'O3',flow_name)
    logger = setup_logger(output=output)

    class p_Omega(Model):
        """Prior space"""
        def __init__(self):
            # Names of parameters to sample
            # self.names = ["H0", "gamma", "kappa", "zp", "alpha", "beta", "mmax", "mmin", "mu_g", "sigma_g", "lambda_peak", "delta_m"]
            
            # Prior bounds for each parameter
            # self.bounds = {"H0":[20,180], "gamma":[0,12], "kappa":[0,6],
            #                "zp":[0,4], "alpha":[1.5,12], "beta":[-4,12],
            #                "mmax":[50,200], "mmin":[2,10], "mu_g":[20,50], "sigma_g":[0.4,10],
            #                "lambda_peak":[0,1], "delta_m":[0,10]}
            self.names = ["H0", "Om0", "w0", "eta"]
            # self.bounds = {'H0':[20,180], 'gamma':[0,12], 'k':[0,6], 'zp':[0,4]} 
            self.bounds = {'H0':[20,180], 'Om0':[0,1], 'w0':[-1.9,0.9], 'eta':[0.5,3.0]}
    
        def log_prior(self, x):
            """
            Returns log of prior given a live point assuming uniform
            priors on each parameter.
            """
            # Check if values are in bounds, returns True/False
            # Then take the log to get 0/-inf and make sure the dtype is float
            log_p = np.log(self.in_bounds(x), dtype="float")
            # Iterate through each parameter (x and y)
            # since the live points are a structured array we can
            # get each value using just the name
            for n in self.names:
                log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
            return log_p
    
        def log_likelihood(self, x):
            """
            Returns log likelihood of given live point
            """
            prior_samples = self.unstructured_view(x).T
            # print(np.shape(x))
            # log_prob = np.zeros(x.size)
            log_prob = np.zeros(1)

            if np.ndim(prior_samples) == 1:
                prior_samples = np.reshape(prior_samples, (flow_class_o3_hlv.hyperparameters['n_conditional_inputs'],1))
            if 'O1' in runs:
                N_events_o1 = len(GW_class.get_event('O1', 'HL'))

                
                log_posterior_o1 = flow_class_o1.temp_funct_post(flow_class_o1, data_o1, prior_samples,
                                                                 N_events_o1, N_theta, N_samples, ndim_target = 5)
                
                
                log_prob += log_posterior_o1
                
            if 'O2' in runs:
                N_evetns_o2_hlv = len(GW_class.get_event('O2', 'HLV'))
                N_evetns_o2_hl = len(GW_class.get_event('O2', 'HL'))
                log_posterior_o2_hlv = flow_class_o2_hlv.temp_funct_post(flow_class_o2_hlv, data_o2_hlv, prior_samples,
                                                                         N_evetns_o2_hlv, N_theta, N_samples, ndim_target = 5)
                log_posterior_o2_hl = flow_class_o2_hl.temp_funct_post(flow_class_o2_hl, data_o2_hl, prior_samples,
                                                                       N_evetns_o2_hl, N_theta, N_samples, ndim_target = 5)
            
                log_prob += log_posterior_o2_hlv + log_posterior_o2_hl
                
            if 'O3' in runs:    
                
                # N_evetns_o3_hl = len(GW_class.get_event('O3', 'HL'))
                # N_evetns_o3_lv = len(GW_class.get_event('O3', 'LV'))
                # N_evetns_o3_hv = len(GW_class.get_event('O3', 'HV'))
                
                log_posterior_o3_hlv = flow_class_o3_hlv.temp_funct_post(flow_class_o3_hlv, data_o3_hlv, prior_samples,
                                                                         N_evetns_o3_hlv, N_theta, N_samples, ndim_target = 14)
                # log_posterior_o3_hl = flow_class_o3_hl.temp_funct_post(flow_class_o3_hl, data_o3_hl, prior_samples, N_evetns_o3_hl,
                #                                                        N_theta, N_samples, ndim_target = 5)
                # log_posterior_o3_lv = flow_class_o3_lv.temp_funct_post(flow_class_o3_lv, data_o3_lv, prior_samples, N_evetns_o3_lv,
                #                                                        N_theta, N_samples, ndim_target = 5)
                # log_posterior_o3_hv = flow_class_o3_hv.temp_funct_post(flow_class_o3_hv, data_o3_hv, prior_samples, N_evetns_o3_hv,
                #                                                        N_theta, N_samples, ndim_target = 5)
                
                log_prob += log_posterior_o3_hlv 
                # log_prob += log_posterior_o3_hl
                # log_prob += log_posterior_o3_lv
                # log_prob += log_posterior_o3_hv
            
            return log_prob
    
        
        
    fs = FlowSampler(p_Omega(), output=output, resume=False, seed=1234, nlive=live_points)
                     # pytorch_threads=2,  # Allow pytorch to use 2 threads
                     # n_pool=2)  # Threads for evaluating the likelihood)
    
    # And go!
    fs.run()

else: 
    raise ValueError('Sampler {} is not implemented or it does not exist'.format(sampler))

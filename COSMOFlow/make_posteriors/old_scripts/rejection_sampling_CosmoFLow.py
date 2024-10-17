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

ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-run", "--run", required=True,
   help="Observational run")
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
run = str(args['run'])
flow_name = str(args['flow_name'])
Ntheta = int(args['Ntheta'])
Nomega = int(args['Nomega'])
Ntotal = int(args['Ntotal'])
device = str(args['device'])

os.chdir("..")
path_flow = 'train_flow/trained_flows_and_curves/'
flow_class = Handle_Flow(path_flow, flow_name, device, epoch = None)
n_conditional = flow_class.hyperparameters['n_conditional_inputs']


if run == 'O1':
    events = [ 'GW150914_095045','GW151226_033853'] # O1
elif run == 'O2':
    events = ['GW170809_082821', 'GW170814_103043', 'GW170818_022509'] #O2
elif run == 'O3':
    events = ['GW190408_181802', 'GW190412_053044', 'GW190503_185404', 'GW190512_180714', 'GW190513_205428',
                      'GW190517_055101', 'GW190519_153544', 'GW190521_030229', 'GW190602_175927', 'GW190701_203306',
                      'GW190720_000836', 'GW190727_060333', 'GW190728_064510', 'GW190828_063405', 'GW190828_065509',
                      'GW190915_235702', 'GW190924_021846', 'GW200129_065458', 'GW200202_154313', 'GW200224_222234',
                      'GW200311_115853'] #O3
Nevents = len(events)
data_total = pd.DataFrame()
for event in tqdm(events): 
    df = utilities.load_data_GWTC(event)
    df = df[['luminosity_distance', 'mass_1', 'mass_2', 'ra', 'dec']].head(Ntheta)
    data = [data_total, df]
    data_total = pd.concat(data)

# cosmoflow/COSMOFlow/train_flow/trained_flows_and_curves/MULTI_PARA_MASS_test_V3/hyperparameters.txt

    

accepted_points = []
ESS_list, N_accept_list = [], []
wi_list = []
samples_proposal = []
time_list = [] 
counter = 0 
probs = [] 


while True:
    start = time.time()
    prior_samples = utilities.make_samples(Nomega)
    
    likelihood_eval = flow_class.p_theta_H0_one_go_batch(data_total, prior_samples.T)
    probs.append(likelihood_eval)
    likelihood_eval = np.reshape(likelihood_eval, (int(Nevents*Ntheta),int(Nomega)))
    posterior_eval = flow_class.get_posterior_from_batch(likelihood_eval, Nomega,  Ntheta , Nevents)
    
    log_w_i_unweighted = np.log(posterior_eval)
    wi_list.append(np.exp(log_w_i_unweighted))
    N = np.sum(np.array(wi_list)/np.max(wi_list))
    ESS = np.sum(np.array(wi_list))**2 / (np.sum(np.array(wi_list)**2))
    samples_proposal.append(prior_samples)
    end = time.time()
    time_list.append(end - start)
    counter += 1
    efficiency = counter * Nomega / N
    sys.stdout.write('\rESS = {} | N = {} | iteration = {} | time = {} | total_cumulative_time = {} | efficiency = {} '.format(round(ESS,2), round(N,2),counter,round(end - start,2), round(np.sum(np.array(time_list)),2),round(efficiency),2))
    
    N_accept_list.append(N)
    plt.plot(np.cumsum(np.array(time_list)), np.array(N_accept_list), '-k', linewidth = 4)
    plt.xlabel(r'time [$s$]', fontsize = 13)
    plt.ylabel(r'$N = <w_{i}> $', fontsize = 13)
    plt.axhline(y = Ntotal,color = 'r', linewidth = 3,  label = 'Ntotal')
    plt.grid(True)
    plt.yscale('log')
    plt.savefig('make_posteriors/posterior_samples_rejection/statistics_rejection_sampling.png', dpi = 300)
    plt.show()
    plt.close()
    
    
    if N >= Ntotal:
        break

times = time_list
weights = np.concatenate(wi_list)
samples_from_prior = np.concatenate(samples_proposal, axis = 1)
N_points = len(weights)
weights /= np.max(weights)
t_i = np.random.uniform(0,1, N_points)
inx_accept = np.where(t_i < weights)[0]
accepted_points = samples_from_prior[:,inx_accept]

dictionary_samples = {'H0':accepted_points[0,:], 'gamma':accepted_points[1,:], 'kappa':accepted_points[2,:], 'zp':accepted_points[3,:],
                     'alpha':accepted_points[4,:], 'beta':accepted_points[5,:], 'mmax':accepted_points[6,:], 'mmin':accepted_points[7,:],
                     'mu_g':accepted_points[8,:], 'sigma_g':accepted_points[9,:], 'lambda_peak':accepted_points[10,:], 'delta_m':accepted_points[11,:], 'weights':weights, 'times':times}

# dictionary_data = { }

# posterior_samples = pd.DataFrame(dictionary_samples)
# posterior_data = pd.DataFrame(dictionary_data)
# posterior_samples.to_csv('make_posteriors/posterior_samples_rejection/run_{}_FLOW_{}_samples.csv'.format(run,flow_name))
with open('make_posteriors/posterior_samples_rejection/run_{}_FLOW_{}_Ntheta_{}_Nomega_{}_Ntotal_{}_data.pickle'.format(run,flow_name,Ntheta, Nomega, Ntotal), 'wb') as handle:
        pickle.dump(dictionary_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

from glasflow.flows import RealNVP, CouplingNSF
from scipy.stats import norm
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import display
from tqdm import tqdm 
import corner
import numpy as np
import pickle
from scipy.special import erf
from cosmology_functions import priors 
from cosmology_functions import cosmology 
import astropy.constants as const
from scipy.stats import norm, gaussian_kde
from tqdm import tqdm 
from scipy.stats import ncx2
import sys
import os
import shutil
import pickle
import bilby
from bilby.core.prior import Uniform
from scipy.stats import entropy
bilby.core.utils.log.setup_logger(log_level=0)
from astropy.coordinates import spherical_to_cartesian, cartesian_to_spherical
from torch import logit, sigmoid

import wandb
wandb.init(project="CosmoFlow", entity="federico_s")


def read_data(batch):
    path_name ="/data/wiay/federico/PhD/cosmoflow/COSMOFlow/data_gwcosmo/galaxy_catalog/training_data_from_MLP/"
    data_name = "NEW_MADAU_batch_{}_500000_N_SNR_11_Nelect_5__Full_para_v2.csv".format(batch)
    GW_data = pd.read_csv(path_name+data_name,skipinitialspace=True, usecols=['H0', 'dl','m1', 'm2','a1', 'a2', 'tilt1', 'tilt2', 
                                                                              'RA', 'dec','theta_jn', 'snr',
                                                                              'phi_12','polarization','geo_time',
                                                                              'phi_jl', 'app_mag', 'inx'])
    return GW_data

list_data = [] 
for i in range(2):
    list_data.append(read_data(i+1))


GW_data = pd.concat(list_data)
print()
print('Preview the data: SIZE = ({},{})'.format(len(GW_data), len(GW_data.columns)))
print()
print((GW_data.head()))   



data = GW_data[['dl','m1', 'm2','a1', 'a2', 'tilt1', 'tilt2', 'RA', 'dec', 'theta_jn', 'phi_jl', 
                                                                             'phi_12', 'polarization', 'geo_time', 'H0']]

#transform Polar into cartesian and spins to sigmoids
def spherical_to_cart(dl, ra, dec):
    
    x,y,z = spherical_to_cartesian(dl, dec, ra)
    return x,y,z

coordinates= data[['dl', 'RA', 'dec']]
dl = np.array(coordinates.dl)
ra = np.array(coordinates.RA)
dec = np.array(coordinates.dec)

x,y,z = spherical_to_cart(dl, ra, dec)

data['xcoord'] = x
data['ycoord'] = y
data['zcoord'] = z




data = data[['xcoord', 'ycoord', 'zcoord', 'm1', 'm2','a1', 'a2', 'tilt1', 'tilt2', 'theta_jn', 'phi_jl', 
                                                                             'phi_12', 'polarization', 'geo_time', 'H0']]

print(data.head(10))

def scale_data(data_to_scale):
    target = data_to_scale[data_to_scale.columns[0:-1]]
    conditioners = np.array(data_to_scale[data_to_scale.columns[-1]]).reshape(-1,1)
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaled_target = scaler_x.fit_transform(target) 
    scaled_conditioners = scaler_y.fit_transform(conditioners)  
    scaled_data = np.hstack((scaled_target, scaled_conditioners))
    scaled_data = pd.DataFrame(scaled_data, index=data_to_scale.index, columns=data_to_scale.columns)
    return scaler_x, scaler_y, scaled_data
    
scaler_x, scaler_y, scaled_data = scale_data(data)
x_train, x_val = train_test_split(scaled_data, test_size=0.25)

batch_size = 25000


train_tensor = torch.from_numpy(np.asarray(x_train).astype('float32'))
val_tensor = torch.from_numpy(np.asarray(x_val).astype('float32'))


X_scale_train = train_tensor[:,:-1]
Y_scale_train = train_tensor[:,-1]
X_scale_val = val_tensor[:,:-1]
Y_scale_val = val_tensor[:,-1]

train_dataset = torch.utils.data.TensorDataset(X_scale_train ,Y_scale_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True)

val_dataset = torch.utils.data.TensorDataset(X_scale_val , Y_scale_val)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                           shuffle=False)


conditional_val = scaler_y.inverse_transform(np.array(Y_scale_val).reshape(-1,1))
target_val = scaler_x.inverse_transform(X_scale_val)




# Define Flow
n_inputs = 14
n_conditional_inputs = 1
n_neurons = 128
n_transforms = 6
n_blocks_per_transform = 4


# best_model = copy.deepcopy(flow.state_dict())
# best_val_loss = np.inf


#Standard Gaussian
g = np.linspace(-5, 5, 1000)
gaussian = norm.pdf(g)




def KL_evaluate(samples):
    def pdf_evaluate(samples):
        density = gaussian_kde(samples)
        kde_points = density.pdf(g)
        return np.array(kde_points)
    pdf_samples = pdf_evaluate(samples)
    return pdf_samples, entropy(pdf_samples, gaussian)

##################################### TRAINING #####################################
#Define device GPU or CPU



#Decay LR
decayRate = 0.999
  

# Loss
loss_dict = dict(train=[], val=[])


#stor KL values at each epoch 
KL_vals1 = []
KL_vals2 = []
KL_vals3 = []
KL_vals4 = []
KL_vals5 = []
KL_vals6 = []
KL_vals7 = []
KL_vals8 = []
KL_vals9 = []
KL_vals10 = []
KL_vals11 = []
KL_vals12 = []
KL_vals13 = []
KL_vals14 = []
# JS_vals6 = []

#W&B

hyp_dict = {'n_neurons': 128, 'n_transforms':4, 'n_blocks':3, 'batch_size':batch_size, 'learning_rate': 0.0003, 'epochs': 1000}




config = hyp_dict

with wandb.init(config = config):
    config = wandb.config
    

    flow = RealNVP(n_inputs= n_inputs,
            n_transforms= config.n_transforms,
            n_neurons= config.n_neurons,
            n_conditional_inputs = n_conditional_inputs,
            n_blocks_per_transform = config.n_blocks,
            batch_norm_between_transforms=True,
            dropout_probability=0.05,
            linear_transform=None)    
    device = 'cuda:2'
    flow.to(device)
    optimiser_adam = torch.optim.Adam(flow.parameters(), lr=config.learning_rate, weight_decay=0)
    my_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser_adam, config.epochs, eta_min=0, last_epoch=- 1, verbose=False) 
    for j in range(config.epochs):

        optimiser = optimiser_adam


        flow.to(device)

        #Train
        train_loss = 0
        for batch in train_loader:

            target_train, conditionals_train = batch 
            flow.train()
            optimiser.zero_grad()
            loss = -flow.log_prob(target_train.to(device)[:,:n_inputs], conditional=conditionals_train.reshape(-1,1).to(device)).cpu().mean()
            loss.backward()

            #optimiser step

            optimiser.step()


            train_loss += loss.item()

        train_loss /= len(train_loader)
        loss_dict['train'].append(train_loss)
        lr_upd = optimiser.param_groups[0]["lr"]
        my_lr_scheduler.step()

        #Validate
        flow.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader: 
                target_val, condtionals_val = batch 


                loss = - flow.log_prob(target_val.to(device)[:,:n_inputs],conditional=condtionals_val.reshape(-1,1).to(device)).cpu().mean()
                val_loss += loss.item()       
        val_loss /= len(val_loader)
        loss_dict['val'].append(val_loss)

#         if val_loss < best_val_loss:
#             best_model = copy.deepcopy(flow.state_dict())
#             best_val_loss = val_loss
    

        flow.eval()
        with torch.no_grad():
            conditionals = Y_scale_val.to(device)
            target_data = X_scale_val.to(device)
            latent_samples, _= flow.forward(target_data[:,:n_inputs], conditional=conditionals.reshape(-1,1))
        z_= latent_samples.cpu().detach().numpy()[0:10000]

        #KDEsdensity.pdf(g)
        kde_points1, kl_val_1 = KL_evaluate(z_[:,0])
#         kde_points2, kl_val_2 = KL_evaluate(z_[:,1])
#         kde_points3, kl_val_3 = KL_evaluate(z_[:,2])
#         kde_points4, kl_val_4 = KL_evaluate(z_[:,3])
#         kde_points5, kl_val_5 = KL_evaluate(z_[:,4])
#         kde_points6, kl_val_6 = KL_evaluate(z_[:,5])
#         kde_points7, kl_val_7 = KL_evaluate(z_[:,6])
#         kde_points8, kl_val_8 = KL_evaluate(z_[:,7])
#         kde_points9, kl_val_9 = KL_evaluate(z_[:,8])
#         kde_points10, kl_val_10 = KL_evaluate(z_[:,9])
#         kde_points11, kl_val_11 = KL_evaluate(z_[:,10])
#         kde_points12, kl_val_12 = KL_evaluate(z_[:,11])
#         kde_points13, kl_val_13 = KL_evaluate(z_[:,12])
#         kde_points14, kl_val_14 = KL_evaluate(z_[:,13])
        
        
        wandb.log({'train_loss':train_loss, 'validation_loss':val_loss, 'kl_divergence':kl_val_1})
        torch.save(flow.state_dict(), os.path.join(wandb.run.dir,'Test_WandB'))
    
    

#     KL_vals1.append(kl_val_1)
#     KL_vals2.append(kl_val_2)
#     KL_vals3.append(kl_val_3)
#     KL_vals4.append(kl_val_4)
#     KL_vals5.append(kl_val_5)
#     KL_vals6.append(kl_val_6)
#     KL_vals7.append(kl_val_7)
#     KL_vals8.append(kl_val_8)
#     KL_vals9.append(kl_val_9)
#     KL_vals10.append(kl_val_10)
#     KL_vals11.append(kl_val_11)
#     KL_vals12.append(kl_val_12)
#     KL_vals13.append(kl_val_13)
#     KL_vals14.append(kl_val_14)

## Real time plotting
#     #Define figure
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25,15))

#     #ax1.set_title('lr = ' + str(lr1))
#     ax1.plot(np.linspace(1,j+1, len(loss_dict['train'])), loss_dict['train'],'k', label='Train')
#     ax1.plot(np.linspace(1,j+1, len(loss_dict['train'])), loss_dict['val'],'r', label='Validation', alpha=0.5)
#     ax1.set_ylabel('loss', fontsize = 20)
#     ax1.set_xlabel('Epochs', fontsize = 20)
#     ax1.set_xscale('log')
#     ax1.set_ylim([-12.5,-5.5])
#     ax1.set_xlim([1,n_epochs])
#     ax1.xaxis.set_tick_params(labelsize=20)
#     ax1.yaxis.set_tick_params(labelsize=20)
#     ax1.grid(True) 
#     ax1.legend(fontsize = 20)

#     #Real time latent space plotting        

#     ax2.set_ylim([0,0.5])
#     ax2.set_xlim([-5,5])
#     ax2.plot(g, kde_points1, linewidth=3,alpha = 0.6, label = r'$z_{1}$')
#     ax2.plot(g, kde_points2, linewidth=3,alpha = 0.6, label = r'$z_{2}$')
#     ax2.plot(g, kde_points3, linewidth=3,alpha = 0.6, label = r'$z_{3}$')
#     ax2.plot(g, kde_points4, linewidth=3,alpha = 0.6, label = r'$z_{4}$')
#     ax2.plot(g, kde_points5, linewidth=3,alpha = 0.6, label = r'$z_{5}$')
#     ax2.plot(g, kde_points6, linewidth=3,alpha = 0.6, label = r'$z_{6}$')
#     ax2.plot(g, kde_points7, linewidth=3,alpha = 0.6, label = r'$z_{7}$')
#     ax2.plot(g, kde_points8, linewidth=3,alpha = 0.6, label = r'$z_{8}$')
#     ax2.plot(g, kde_points9, linewidth=3,alpha = 0.6, label = r'$z_{9}$')
#     ax2.plot(g, kde_points10, linewidth=3,alpha = 0.6, label = r'$z_{10}$')
#     ax2.plot(g, kde_points11, linewidth=3,alpha = 0.6, label = r'$z_{11}$')
#     ax2.plot(g, kde_points12, linewidth=3,alpha = 0.6, label = r'$z_{12}$')
#     ax2.plot(g, kde_points13, linewidth=3,alpha = 0.6, label = r'$z_{13}$')
#     ax2.plot(g, kde_points14, linewidth=3,alpha = 0.6, label = r'$z_{14}$')
# #     ax2.plot(g, kde_points6, linewidth=3, label = r'$z_{5}$')
#     ax2.plot(g, gaussian,linewidth=5,c='k',label=r'$\mathcal{N}(0;1)$')

#     ax2.legend(fontsize = 15)
#     ax2.grid(True)
#     ax2.set_ylabel('$p(z)$',fontsize = 20)
#     ax2.set_xlabel('$z$',fontsize = 20)
#     ax2.xaxis.set_tick_params(labelsize=20)
#     ax2.yaxis.set_tick_params(labelsize=20)

#     #Real time JS div between gaussian and latent 
#     ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), KL_vals1,linewidth=3,alpha = 0.6, label = r'$z_{l}$')
#     ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), KL_vals2,linewidth=3,alpha = 0.6,  label = r'$z_{2}$')
#     ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), KL_vals3,linewidth=3,alpha = 0.6,  label = r'$z_{3}$')
#     ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), KL_vals4,linewidth=3,alpha = 0.6,  label = r'$z_{4}$')
#     ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), KL_vals5,linewidth=3,alpha = 0.6,  label = r'$z_{5}$')
#     ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), KL_vals6,linewidth=3,alpha = 0.6, label = r'$z_{6}$')
#     ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), KL_vals7,linewidth=3,alpha = 0.6,  label = r'$z_{7}$')
#     ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), KL_vals8,linewidth=3,alpha = 0.6,  label = r'$z_{8}$')
#     ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), KL_vals9,linewidth=3,alpha = 0.6,  label = r'$z_{9}$')
#     ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), KL_vals10,linewidth=3,alpha = 0.6,  label = r'$z_{10}$')
#     ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), KL_vals11,linewidth=3,alpha = 0.6,  label = r'$z_{11}$')
#     ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), KL_vals12,linewidth=3,alpha = 0.6,  label = r'$z_{12}$')
#     ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), KL_vals13,linewidth=3,alpha = 0.6,  label = r'$z_{13}$')
#     ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), KL_vals14,linewidth=3,alpha = 0.6,  label = r'$z_{14}$')
# #     ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), JS_vals6,linewidth=3,alpha = 0.6,  label = r'$z_{5}$')

#     ax3.set_ylabel('KLDiv', fontsize = 20)
#     ax3.set_xlabel(r'Epochs', fontsize = 20)
#     ax3.set_yscale('log')
#     ax3.set_xscale('log')
#     ax3.set_xlim([1,n_epochs])
#     ax3.grid(True)
#     ax3.xaxis.set_tick_params(labelsize=20)
#     ax3.yaxis.set_tick_params(labelsize=20)
#     ax3.legend(fontsize = 15)
#     fig.tight_layout()
#     fig.savefig(path+folder_name+'/training.png', dpi = 50 )
        
#     plt.close('all')   # Clear figure




#     sys.stdout.write('\rEPOCH = {} || Training Value = {} || Validation Value = {}  '.format(j+1,round(loss_dict['train'][-1], 5), round(loss_dict['val'][-1], 5)))
#     sys.stdout.flush()


# flow.load_state_dict(best_model)

# print()
# print('Saving FLOW model in {}'.format(path+folder_name))
# print()
# #Save flow model 
# torch.save(flow.state_dict(), path+folder_name+'/flow.pt')

# #save model hyperparameters
# para = {'batch_size': batch_size,
#           'n_epochs': n_epochs,
#           'shuffle': True,
#           'activation': None,
#           'dropout': dp,
#           'learning_rate': lr,
#           'optimizer': 'Adam',
#           'linear_transform':linear_transform,
#           'n_neurons': int(n_neurons),
#           'n_transforms': int(n_transforms),
#           'n_blocks_per_transform': int(n_blocks_per_transform),
#           'n_inputs': int(n_inputs),
#           'n_conditional_inputs':int(n_conditional_inputs),
#           'flow_type': flow_type
#           }

# f = open(path+folder_name+'/hyperparameters.txt','w')
# f.write(str(para))
# f.close()






# # ##################################### TESTING #####################################

# # ###### TEST 1: PP-Plot ######


# print()
# print('TEST 1: KS-test by plotting a PP plot')
# print()



# print()
# print('Making Probability-Probability plot with Validation data')
# print()

# def Flow_samples(conditional, n):
    
#     Y_H0_conditional = scaler_y.transform(conditional.reshape(-1,1))

    
#     conditional = np.array(Y_H0_conditional)
#     data = np.array(conditional)
#     data_scaled = torch.from_numpy(data.astype('float32'))
    
#     flow.eval()
#     flow.to('cpu')
    

#     with torch.no_grad():
#         samples = flow.sample(n, conditional=data_scaled.to('cpu'))
#         samples= scaler_x.inverse_transform(samples.to('cpu'))
#     return samples 



# np.random.seed(1234)
# Nresults =200
# Nruns = 1
# labels = ['x','y', 'z','m1', 'm2','a1', 'a2', 'tilt1','tilt2', 'theta_jn', 'phi_jl', 
#                                                                              'phi_12', 'psi', 'time']
# priors = {}
# for jj in range(14):
#     priors.update({f"{labels[jj]}": Uniform(0, 1, f"{labels[jj]}")})




# for x in range(Nruns):
#     results = []
#     for ii in tqdm(range(Nresults)):
#         posterior = dict()
#         injections = dict()
#         i = 0 
#         for key, prior in priors.items():

#             inx = np.random.randint(len(Y_scale_val))  
#             truths=  scaler_x.inverse_transform(X_scale_val[inx,:].reshape(1,-1))[0]
#             conditional_sample = scaler_y.inverse_transform(Y_scale_val[inx].reshape(1,-1))[0]
#             conditional_sample = conditional_sample *np.ones(10000)
#             samples = Flow_samples(conditional_sample, 10000)
#             posterior[key] = samples[:,i] 
#             injections[key] = truths[i].astype('float32').item()
#             i += 1

#         posterior = pd.DataFrame(dict(posterior))
#         result = bilby.result.Result(
#             label="test",
#             injection_parameters=injections,
#             posterior=posterior,
#             search_parameter_keys=injections.keys(),
#         priors = priors )
#         results.append(result)

#     fig = bilby.result.make_pp_plot(results, filename=path+folder_name+'/PP',
#                               confidence_interval=(0.68, 0.90, 0.99, 0.9999))






# ##### TEST 2: Resample target data #####
# print()
# print('TEST 2: resampling the data space by by applying the backwards fuction from latent space to data space')
# print()

# N = 150000

# combined_samples = []
# total_H0_samples = []
# while True: 
    
#     H0_samples = np.random.uniform(20,120,N)

#     samples = Flow_samples(H0_samples, N)


#     #Condtions 
    
# #     z = cosmology.fast_dl_to_z_v2(samples[:,0], H0_samples)
    
# #     samples = np.concatenate([samples, H0_samples, z], axis=1)
# #     samples = samples[np.where(samples[:,0] > 0)[0], :]
# #     samples = samples[np.where(samples[:,1] > 0)[0], :]
# #     samples = samples[np.where(samples[:,2] > 0)[0], :]
# #     samples = samples[np.where((samples[:,3] > 0) & (samples[:,3] <= 0.99))[0], :]
# #     samples = samples[np.where((samples[:,4] > 0) & (samples[:,4] <= 0.99))[0], :]
# #     samples = samples[np.where((samples[:,5] > 0) & (samples[:,5] <= np.pi))[0], :]
# #     samples = samples[np.where((samples[:,6] > 0) & (samples[:,6] <= np.pi))[0], :]
# #     samples = samples[np.where((samples[:,7] > 0) & (samples[:,7] <= 2*np.pi))[0], :]
# #     samples = samples[np.where((samples[:,8] >= -np.pi/2) & (samples[:,8] <= np.pi/2))[0], :]
# #     samples = samples[np.where((samples[:,9] >= 0) & (samples[:,9] <= np.pi))[0], :]
# #     m1 = (1/(1+a)) * samples[:,1]
# #     m2 = (1/(1+a)) * samples[:,2]
# #     sumM = m1 + m2 
# #     indicies = np.where(sumM <= 100)[0]
# #     samples = samples[indicies,:]

#     combined_samples = combined_samples + list(samples)

    
#     if len(np.array(combined_samples)) >= N:
#         combined_samples = np.array(combined_samples)[:N,:]
#         break




# c1 = corner.corner(combined_samples, plot_datapoints=False, smooth = True, levels = (0.5, 0.9), color = 'red', hist_kwargs = {'density' : 1})
# fig = corner.corner(data[['xcoord', 'ycoord', 'zcoord', 
#                           'm1', 'm2','a1_logit',
#                           'a2_logit', 'tilt1', 'tilt2',
#                           'theta_jn', 'phijl_logit', 'phi12_logit',
#                           'pol_logit', 'tc_logit']] , plot_datapoints=False, smooth = True, fig = c1, levels = (0.5, 0.9), plot_density=True,labels=[r'$x[Mpc]$',r'$y[Mpc]$',r'$z[Mpc]$', r'$m_{1,z}$', r'$m_{2,z}$',r'$loga_{1}$', r'$loga_{2}$', r'$tilt_{1}$', r'$tilt_{2}$', r'$\theta_{JN}$', r'$log\phi_{JL}$',  r'$log\phi_{12}$',r'$log\psi$', r'$logt_{geo}$'], hist_kwargs = {'density' : 1})



# plt.savefig(path+folder_name+'/flow_resample.png', dpi = 100)




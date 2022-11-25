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
import argparse
import wandb
wandb.init(project="CosmoFlow", entity="federico_s")
#For use 
# python3 train_flow_v3.py -Name Flow_Glade_mth_v1-batch 25000 -train_size 0.8 -flow_type RealNVP -epochs 10 -neurons 128 -layers 6 -nblock 4 -lr 0.001



#pass arguments 
# Construct the argument parser
# ap = argparse.ArgumentParser()

# # Add the arguments to the parser
# ap.add_argument("-Name", "--Name_folder", required=True,
#    help="Name of the folder to save the FLOW model")
# ap.add_argument("-data", "--data_folder", required=False,
#    help="Name of the folder where training data is stored")
# ap.add_argument("-batch", "--batch_size", required=False,
#    help="batch size of the data to pass", default = 5000)
# ap.add_argument("-train_size", "--train_size", required=False,
#    help="size of trainign data percentage", default = 0.75)
# ap.add_argument("-flow_type", "--flow_type", required=False,
#    help="Number of training iterations", default = "RealNVP")
# ap.add_argument("-epochs", "--epochs", required=False,
#    help="Number of training iterations", default = 1000)
# ap.add_argument("-neurons", "--neurons", required=False,
#    help="Number of neurons in layer", default = 12)
# ap.add_argument("-layers", "--layers", required=False,
#    help="Number of hidden layers", default = 4)
# ap.add_argument("-nblock", "--nblock", required=False,
#    help="Neurons of blocks", default = 2)
# ap.add_argument("-lr", "--learning_rate", required=False,
#    help="learning_rate", default = 0.005)
# ap.add_argument("-lr_scheduler", "--learning_rate_scheduler", required=False,
#    help="learning_rate", default = None)
# ap.add_argument("-linear_transform", "--linear_transform", required=False,
#    help="type of transformation", default = None)
# ap.add_argument("-drop_out", "--drop_out", required=False,
#    help="drop_out", default = 0.0)
# ap.add_argument("-log_it", "--log_it", required=False,
#    help="logit the uniform distributions", default = 0)


flow_type = 'RealNVP'
batch = int(25_000)
train_size = float(0.8)
neurons = int(128)
layers = int(6)
nblock = int(3)
lr = float(0.003)
lr_scheduler = 'CosineAnnealingLR'
epochs = int(10)
dp = float(0.05)
log_it = int(1)
lt = 'lu'



hyp_dict = {'flow_type' : flow_type,
            'batch_size' : batch,
            'n_neurons': neurons, 
            'n_transforms':layers, 
            'n_blocks':nblock, 
            'learning_rate': lr, 
            'epochs': epochs,
            'drop_out': dp,
            'linear_transform':lt}



config = hyp_dict

with wandb.init(config = config):
    config = wandb.config



    def read_data(batch):
        path_name ="/data/wiay/federico/PhD/cosmoflow/COSMOFlow/data_gwcosmo/galaxy_catalog/training_data_from_MLP/"
        data_name = "name_training_data_catalog_True_band_Bj_batch_{}_N_250000_SNR_11_Nelect_5__Full_para_v1.csv".format(batch)
        GW_data = pd.read_csv(path_name+data_name,skipinitialspace=True, usecols=['H0', 'dl','m1', 'm2','a1', 'a2', 'tilt1', 
                                                                                  'tilt2', 'RA', 'dec', 'theta_jn', 'phi_jl', 
                                                                                 'phi_12', 'polarization', 'geo_time'])
        return GW_data

    list_data = [] 
    for i in range(2):
        list_data.append(read_data(i+1))


    GW_data = pd.concat(list_data)
    GW_data = GW_data.drop_duplicates(keep='first').sample(frac=1)
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

    data.loc[:,'xcoord'] = x
    data.loc[:,'ycoord'] = y
    data.loc[:,'zcoord'] = z

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



    data = data[['xcoord', 'ycoord', 'zcoord', 'm1', 'm2','a1', 'a2', 'tilt1', 'tilt2', 'theta_jn', 'phi_jl', 
                                                                                 'phi_12', 'polarization', 'geo_time', 'H0']]


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

    if config.log_it == 1:
        logit_data(scaled_data)
        scaled_data = scaled_data[np.isfinite(scaled_data).all(1)]

    x_train, x_val = train_test_split(scaled_data, test_size=float(1 - train_size))

    batch_size=batch

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

    path = '/data/wiay/federico/PhD/cosmoflow/COSMOFlow/train_flow/NEW_MADAU_scalers'

    print()
    print('Saving Scalers X and Y')
    print()
    scalerfileX = path+'/scaler_x.sav'
    scalerfileY = path+'/scaler_y.sav'
    pickle.dump(scaler_x, open(scalerfileX, 'wb'))
    pickle.dump(scaler_y, open(scalerfileY, 'wb'))



    print()
    print('Batch size = {}'.format(config.batch_size))
    print('Train/Validation = {}% / {}%'.format(train_size * 100, round(1- train_size, 2)*100))
    print()


    # Define Flow
    n_inputs = 14
    n_conditional_inputs = 1
    n_neurons = config.n_neurons
    n_transforms = config.n_transforms
    n_blocks_per_transform = config.n_blocks


    print()
    print('n_inputs = {} ; n_conditional_inputs = {}'.format(n_inputs, n_conditional_inputs))
    print('NEURONS = {}; LAYERS = {}; BLOCKS = {}'.format(n_neurons, n_transforms, n_blocks_per_transform))
    print()



    if config.flow_type == 'RealNVP':
        if config.linear_transform == 0:
            flow = RealNVP(n_inputs= n_inputs,
                    n_transforms= config.n_transforms,
                    n_neurons= config.n_neurons,
                    n_conditional_inputs = n_conditional_inputs,
                    n_blocks_per_transform = config.n_blocks,
                    batch_norm_between_transforms=True,
                    dropout_probability=config.drop_out,
                    linear_transform=None)
        else:
            flow = RealNVP(n_inputs= n_inputs,
                    n_transforms= config.n_transforms,
                    n_neurons= config.n_neurons,
                    n_conditional_inputs = n_conditional_inputs,
                    n_blocks_per_transform = config.n_blocks,
                    batch_norm_between_transforms=True,
                    dropout_probability=config.drop_out,
                    linear_transform=config.linear_transform)
            

    elif config.flow_type == 'CouplingNSF':
        if config.linear_transform == 0:
            flow = CouplingNSF(n_inputs= n_inputs,
                    n_transforms= config.n_transforms,
                    n_neurons= config.n_neurons,
                    n_conditional_inputs = n_conditional_inputs,
                    n_blocks_per_transform = config.n_blocks,
                    batch_norm_between_transforms=True,
                    dropout_probability=config.drop_out,
                    linear_transform=None)
        else:      
            flow = CouplingNSF(n_inputs= n_inputs,
                    n_transforms= config.n_transforms,
                    n_neurons= config.n_neurons,
                    n_conditional_inputs = n_conditional_inputs,
                    n_blocks_per_transform = config.n_blocks,
                    batch_norm_between_transforms=True,
                    dropout_probability=config.drop_out,
                    linear_transform=config.linear_transform)
    else: 
        raise ValueError('Flow not implemented')




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
    device = 'cuda:2'
    flow.to(device)

    #learning rate
    lr = config.learning_rate
    print('LEARNING RATE = {}'.format(lr))

    #Adam optimiser
    optimiser_adam = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=0)

    #Epochs
    n_epochs = config.epochs
    print('EPOCHS = {}'.format(n_epochs))

    #Decay LR
    decayRate = 0.999

    if lr_scheduler == 'ExponentialLR':

        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser_adam, gamma=decayRate)

    elif lr_scheduler == 'CyclicLR':

        my_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimiser_adam, base_lr=0.01, max_lr=0.03, step_size_up=75)   

    elif lr_scheduler == 'CosineAnnealingLR':

        my_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser_adam, config.epochs, eta_min=0, last_epoch=- 1, verbose=False)   





    # Loss and kl
   
    loss_dict = dict(train=[], val=[], kl=[])
    
    
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



        flow.eval()
        with torch.no_grad():
            conditionals = Y_scale_val.to(device)
            target_data = X_scale_val.to(device)
            latent_samples, _= flow.forward(target_data[:,:n_inputs], conditional=conditionals.reshape(-1,1))
        z_= latent_samples.cpu().detach().numpy()[0:10000]
        
        
        temp = 0
        for t in range(14):
            kde_points, kl_val = KL_evaluate(z_[:,t])
            temp += kl_val 
        kl_avg_val = float(temp/14)
        loss_dict['kl'].append(kl_avg_val)


    

        sys.stdout.write('\rEPOCH = {} || Training Value = {} || Validation Value = {} || Kullback–Leibler divergence = {} '.format(j+1,round(loss_dict['train'][-1], 5),
                                                                                                   round(loss_dict['val'][-1], 5),
                                                                                                  round(loss_dict['kl'][-1], 5)))
        sys.stdout.flush()

        wandb.log({'train_loss':train_loss, 'validation_loss':val_loss, 'kl_divergence':kl_avg_val})
        torch.save(flow.state_dict(), os.path.join(wandb.run.dir,'flow_v1_sweep.pt'))








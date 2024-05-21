import sys
sys.path.append("..")
from cosmology_functions import utilities
from gw_functions.gw_event_names import GW_events
sys.path.append("train_flow")

import numpy as np
import pandas as pd
import os 
import shutil
from glasflow.flows import RealNVP, CouplingNSF
import copy
from scipy.stats import norm
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import pickle
import corner

parameters = {'population':'BBH'}

gwevents = GW_events(parameters)

O3_evetns_HLV = gwevents.get_event('O3', 'HLV')
O3_evetns_HL = gwevents.get_event('O3', 'HL')
O3_evetns_LV = gwevents.get_event('O3', 'LV')
O3_evetns_HV = gwevents.get_event('O3', 'HV')
# O2_evetns_HLV = gwevents.get_event('O2', 'HLV')
# O2_evetns_HL = gwevents.get_event('O2', 'HL')
# O1_evetns_HL = gwevents.get_event('O1', 'HL')

# list_events = [O1_evetns_HL, O2_evetns_HLV, O2_evetns_HL, O3_evetns_HLV, O3_evetns_HL, O3_evetns_LV, O3_evetns_HV ] 
list_events = [O3_evetns_HLV, O3_evetns_HL, O3_evetns_LV, O3_evetns_HV ] 
list_events = np.concatenate(list_events)
# list_events = ["GW190513_205428"] 
for event in list_events:
    if event == "GW190513_205428":
        continue
    print('Making cookie cutter for {}'.format(event))
    df = gwevents.load_data_GWTC(event)[['luminosity_distance', 'mass_1', 'mass_2', 'ra', 'dec']]
    df = df.dropna()

    n_inputs = 5
    n_transforms = 3
    n_neurons = 12
    n_conditional_inputs = 0
    n_blocks_per_transform = 2
    device = 'cuda:0'
    lr = 0.001
    epochs = 100
    lr_scheduler = 'No'
    
    #Save model in folder
    path = 'trained_flows_and_curves/'
    folder_name = event
    
    #check if directory exists
    if os.path.exists(path+folder_name):
        #if yes, delete directory
        shutil.rmtree(path+folder_name)
    
    os.mkdir(path+folder_name)

    
    n_conditionals = 0
    train_size = 0.8
    Scaler = 'Standard'
    batch_size = 2000
    
    def scale_data(data_to_scale, Scaler):
        target = data_to_scale
        if Scaler == 'MinMax':
            scaler_x = MinMaxScaler()
        elif Scaler == 'Standard':
            scaler_x = StandardScaler()
        scaled_target = scaler_x.fit_transform(target) 
    
        # scaled_data = np.hstack((scaled_target))
        # print(np.shape(scaled_target))
        scaled_data = pd.DataFrame(scaled_target, index=data_to_scale.index, columns=data_to_scale.columns)
        return scaler_x, scaled_data
    
    
    scaler_x, scaled_data = scale_data(df, Scaler)
    print(scaled_data)
    #split the data in training and validation data sets 
    x_train, x_val = train_test_split(scaled_data, test_size = 1 - train_size, train_size = train_size)
    
    train_tensor = torch.from_numpy(np.asarray(x_train).astype('float32')) #get training data tensor (torch) 
    val_tensor = torch.from_numpy(np.asarray(x_val).astype('float32')) #get validationsor (torch) 
    
    X_scale_train = train_tensor #get target data for training 
    X_scale_val = val_tensor #get target data for validation
     
    
    train_dataset = torch.utils.data.TensorDataset(X_scale_train) #put both target and conditional data sets into one torch tensor 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True) #initialize data loader 
    
    val_dataset = torch.utils.data.TensorDataset(X_scale_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                               shuffle=False) #initialize dataloader for the validation data set 
    
    target_val = scaler_x.inverse_transform(X_scale_val)
    
    
   
    scalerfileX = path+folder_name+'/scaler_x.sav'
    pickle.dump(scaler_x, open(scalerfileX, 'wb'))







    flow = CouplingNSF(n_inputs= n_inputs,
            n_transforms= n_transforms,
            n_neurons= n_neurons,
            n_conditional_inputs = n_conditional_inputs,
            n_blocks_per_transform = n_blocks_per_transform,
            batch_norm_between_transforms=True,
            dropout_probability=0.0,
            linear_transform='lu')
    
    
    #save model hyperparameters
    para = {'batch_size': batch_size,
              'n_epochs': epochs,
              'shuffle': True,
              'activation': None,
              'dropout': 0.0,
              'learning_rate': lr,
              'optimizer': 'Adam',
              'linear_transform':'lu',
              'n_neurons': int(n_neurons),
              'n_transforms': int(n_transforms),
              'n_blocks_per_transform': int(n_blocks_per_transform),
              'n_inputs': int(n_inputs),
              'n_conditional_inputs':int(n_conditional_inputs),
              'flow_type': 'CouplingNSF',
              'scaler': Scaler,
              'lr_scheduler': lr_scheduler  
              }
    
    f = open(path+folder_name+'/hyperparameters.txt','w')
    f.write(str(para))
    f.close()
    # save best model (starting np.inf of val loss)
    best_model = copy.deepcopy(flow.state_dict())
    best_val_loss = np.inf
    
    #make a Standard Gaussian
    g = np.linspace(-5, 5, 1000)
    gaussian = norm.pdf(g)
    
    ##################################### TRAINING #####################################
    #Define device GPU or CPU
    flow.to(device) #send flow model to device to be trained in
    
    #learning rate
    print('LEARNING RATE = {}'.format(lr))
    
    #Adam optimiser
    optimiser_adam = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=0)
    
    #Epochs
    n_epochs = epochs
    print('EPOCHS = {}'.format(n_epochs))
    
    
    # Loss and kl dictionaries
    loss_dict = dict(train=[], val=[])
    kl_dict =  dict(KL_vals1 = [], KL_vals2 = [], KL_vals3 = [], KL_vals4 = [], KL_vals5 = [], KL_vals6 = [], KL_vals7 = [], KL_vals8 = [])
    
    
    for j in range(n_epochs):
        
        optimiser = optimiser_adam
    
        flow.to(device)
    
        #Train
        train_loss = 0
        for batch in train_loader:
            #get target and target and conditional training data from batch 
            target_train = batch[0]
            
            flow.train() #set flow in train mode 
            optimiser.zero_grad() # set derivatives to zero 
            loss = -flow.log_prob(target_train.to(device)[:,:n_inputs]).cpu().mean() #compute loss
            loss.backward() #compute dloss/dx for every parameter with requires_grad=True (i.e. x.grad += dloss/dx)
            
            #optimiser step
            optimiser.step() #perform one optimiser step 
            train_loss += loss.item() #add the training losses from each batch 
    
        train_loss /= len(train_loader) #divide by the number of batches 
        loss_dict['train'].append(train_loss) #apprned training loss value 
    
        if lr_scheduler != 'No': #if a scheduler was sset up
            lr_upd = optimiser.param_groups[0]["lr"]
            my_lr_scheduler.step() #take a step wit the scheduler 
    
        #Validate
        flow.eval()#set flow in evaluation mode 
        with torch.no_grad(): #freeze all gradients 
            val_loss = 0 #initial validation
            for batch in val_loader: 
                target_val = batch[0] #get validation data fro each batch 
                loss = - flow.log_prob(target_val.to(device)[:,:n_inputs]).cpu().mean() #perform loss function for validatio ndata set 
                val_loss += loss.item() #add validation values from each batch 
        val_loss /= len(val_loader) #get mean validation value 
        loss_dict['val'].append(val_loss) #append validation value 
        
        if val_loss < best_val_loss: #if validation loss is better than the previous step, save new model 
            best_model = copy.deepcopy(flow.state_dict())
            best_val_loss = val_loss
        
        with torch.no_grad():
            #Latent space 
            target_data = X_scale_val.to(device)
            latent_samples, _= flow.forward(target_data) #get samples from latent sapce of flow
            
        z_= latent_samples.cpu().detach().numpy()[0:10000] #10000 samples from latent space 
        
        # z_[np.isnan(z_)] = 1 
        # z_[np.isinf(z_)] = 1 
        # print(z_)
        for i in range(n_inputs):
            
            _, kl_vals = utilities.KL_evaluate_gaussian(z_[:,i], gaussian, g) #evaluate KL between one dimension of the latent space and a gaussian 
            kl_dict[[*kl_dict][i]].append(kl_vals) #save KL valu fro each entry 
        
        
        with open(path+folder_name+'/loss_data.pickle', 'wb') as handle:
            pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path+folder_name+'/kl_data.pickle', 'wb') as handle:
            pickle.dump(kl_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        sys.stdout.write('\rEPOCH = {} || Training Value = {} || Validation Value = {}  '.format(j+1,round(loss_dict['train'][-1], 5), round(loss_dict['val'][-1], 5)))
        sys.stdout.flush()
        
    #save flow model 
    flow.load_state_dict(best_model)
    
    #Save flow model 
    torch.save(flow.state_dict(), path+folder_name+'/flow.pt')

    
    ########### Plot corner plot for comparison
    N = 100000
    samples = flow.sample(N).cpu().detach()
    samples= scaler_x.inverse_transform(np.array(samples))
    
    dict_rand = {'luminsoity_distance':list(samples[:,0]), 'mass_1':list(samples[:,1]), 'mass_2':list(samples[:,2]),  'ra':list(samples[:,3]),  'dec':list(samples[:,4])}
    samples = pd.DataFrame(dict_rand)
    
    
    
    c1 = corner.corner(samples, bins = 50,plot_density=False, plot_datapoints=False, smooth = True, 
                       levels = (0.5, 0.99), color = 'red', hist_kwargs = {'density' : 1}, hist_bin_factor=5)
    
    #data = logit_data(data)
    fig = corner.corner(df[['luminosity_distance',  
                              'mass_1', 'mass_2', 'ra', 'dec']] , bins = 50,
                        plot_datapoints=False, 
                        plot_density=False,
                        smooth = True, 
                        fig = c1, 
                        hist_bin_factor=5,
                        levels = (0.5, 0.99),
                        labels = [r'$D_{L}[Mpc]$',r'$m_{1,z}$', r'$m_{2,z}$', r'$RA$', r'$\delta$'],
                        label_kwargs={'fontsize': 25},
                        hist_kwargs = {'density' : 1})
    plt.savefig(path+folder_name+'/comparison.png', dpi = 300)
    
    plt.show()

    
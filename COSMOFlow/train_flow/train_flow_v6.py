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

#For use 
# python3 train_flow_v5.py -Name Flow_Glade_mth_v1-batch 25000 -train_size 0.8 -flow_type RealNVP -epochs 10 -neurons 128 -layers 6 -nblock 4 -lr 0.001



#pass arguments 
# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-Name", "--Name_folder", required=True,
   help="Name of the folder to save the FLOW model")
ap.add_argument("-data", "--data_folder", required=False,
   help="Name of the folder where training data is stored")
ap.add_argument("-batch", "--batch_size", required=False,
   help="batch size of the data to pass", default = 5000)
ap.add_argument("-train_size", "--train_size", required=False,
   help="size of trainign data percentage", default = 0.75)
ap.add_argument("-flow_type", "--flow_type", required=False,
   help="Number of training iterations", default = "RealNVP")
ap.add_argument("-epochs", "--epochs", required=False,
   help="Number of training iterations", default = 1000)
ap.add_argument("-neurons", "--neurons", required=False,
   help="Number of neurons in layer", default = 12)
ap.add_argument("-layers", "--layers", required=False,
   help="Number of hidden layers", default = 4)
ap.add_argument("-nblock", "--nblock", required=False,
   help="Neurons of blocks", default = 2)
ap.add_argument("-lr", "--learning_rate", required=False,
   help="learning_rate", default = 0.005)
ap.add_argument("-lr_scheduler", "--learning_rate_scheduler", required=False,
   help="learning_rate", default = None)
ap.add_argument("-linear_transform", "--linear_transform", required=False,
   help="type of transformation", default = None)
ap.add_argument("-drop_out", "--drop_out", required=False,
   help="drop_out", default = 0.0)
ap.add_argument("-log_it", "--log_it", required=False,
   help="logit the uniform distributions", default = 0)
ap.add_argument("-device", "--device", required=True,
   help="Which CUDA device [:0 , :1, :2]", default = 'cuda:1')




args = vars(ap.parse_args())
Name = str(args['Name_folder'])
data = str(args['data_folder'])
flow_type = str(args['flow_type'])
batch = int(args['batch_size'])
train_size = float(args['train_size'])
neurons = int(args['neurons'])
layers = int((args['layers']))
nblock = int(args['nblock'])
lr = float(args['learning_rate'])
lr_scheduler = str(args['learning_rate_scheduler'])
epochs = int(args['epochs'])
dp = float(args['drop_out'])
log_it = int(args['log_it'])
device = str(args['device'])


linear_transform = args['linear_transform']
if linear_transform is not None:
    linear_transform = str(args['linear_transform'])
elif  float(linear_transform) == 0:
    linear_transform = 0

print()
print('Name model = {}'.format(Name))
print('data name = {}'.format(data))
print('batch = {}'.format(batch))
print('train size = {}'.format(train_size))
print('neurons = {}'.format(neurons))
print('layers = {}'.format(layers))
print('nblocks = {}'.format(nblock))
print('lr = {}'.format(lr))
print('lr_scheduler = {}'.format(lr_scheduler))
print('linear transform = {}'.format(linear_transform))
print('drop out = {}'.format(dp))
print('device = {}'.format(device))
print()


folder_name = str(Name)
path = 'trained_flows_and_curves/'


#check if directory exists
if os.path.exists(path+folder_name):
    #if yes, delete directory
    shutil.rmtree(path+folder_name)


#Save model in folder
os.mkdir(path+folder_name)
os.mkdir(path+folder_name+'/flows_epochs')

os.chdir('..')


def read_data(batch):
    path_name ="data_gwcosmo/galaxy_catalog/training_data_from_MLP/"
    data_name = "run_O2_name_test_catalog_smooth_v1_catalog_True_band_K_batch_{}_N_100000_SNR_11_Nelect_10__Full_para_v1.csv".format(batch)
    GW_data = pd.read_csv(path_name+data_name,skipinitialspace=True, usecols=['H0', 'luminosity_distance', 
                                                                              'mass_1', 'mass_2', 'ra', 'dec',
                                                                               'theta_jn','psi','geocent_time'])
    return GW_data

list_data = [] 
for i in range(3):
    list_data.append(read_data(i+1))


GW_data = pd.concat(list_data)
print()
print('Preview the data: SIZE = ({},{})'.format(len(GW_data), len(GW_data.columns)))
print()
print((GW_data.head()))   


data = GW_data[['luminosity_distance','mass_1', 'mass_2',
                'ra', 'dec', 'theta_jn', 'psi', 'geocent_time',
                 'H0']]

#transform Polar into cartesian and spins to sigmoids
def spherical_to_cart(dl, ra, dec):
    
    x,y,z = spherical_to_cartesian(dl, dec, ra)
    return x,y,z

coordinates= data[['luminosity_distance', 'ra', 'dec']]
dl = np.array(coordinates.luminosity_distance)
ra = np.array(coordinates.ra)
dec = np.array(coordinates.dec)

x,y,z = spherical_to_cart(dl, ra, dec)

data.loc[:,'xcoord'] = x
data.loc[:,'ycoord'] = y
data.loc[:,'zcoord'] = z

def convert_gps_sday(gps):
    return gps%86164.0905

data['geocent_time'] = convert_gps_sday(data['geocent_time'])

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



data = data[['xcoord', 'ycoord', 'zcoord', 'mass_1', 'mass_2', 'H0']]

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

if log_it == 1:
    logit_data(scaled_data)
    scaled_data = scaled_data[np.isfinite(scaled_data).all(1)]

print(scaled_data)
x_train, x_val = train_test_split(scaled_data, test_size = 1 - train_size, train_size = train_size)

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

path = 'train_flow/trained_flows_and_curves/'

print()
print('Saving Scalers X and Y')
print()
scalerfileX = path+folder_name+'/scaler_x.sav'
scalerfileY = path+folder_name+'/scaler_y.sav'
pickle.dump(scaler_x, open(scalerfileX, 'wb'))
pickle.dump(scaler_y, open(scalerfileY, 'wb'))



print()
print('Train loader size = {}'.format(len(train_loader)))
print('Batch size = {}'.format(batch_size))
print('Train/Validation = {}% / {}%'.format(train_size * 100, round(1- train_size, 2)*100))
print()


# Define Flow
n_inputs = 5
n_conditional_inputs = 1
n_neurons = neurons
n_transforms = layers
n_blocks_per_transform = nblock


print()
print('n_inputs = {} ; n_conditional_inputs = {}'.format(n_inputs, n_conditional_inputs))
print('NEURONS = {}; LAYERS = {}; BLOCKS = {}'.format(n_neurons, n_transforms, n_blocks_per_transform))
print()



if flow_type == 'RealNVP':
    flow = RealNVP(n_inputs= n_inputs,
            n_transforms= n_transforms,
            n_neurons= n_neurons,
            n_conditional_inputs = n_conditional_inputs,
            n_blocks_per_transform = n_blocks_per_transform,
            batch_norm_between_transforms=True,
            dropout_probability=dp,
            linear_transform=linear_transform)

elif flow_type == 'CouplingNSF':
    flow = CouplingNSF(n_inputs= n_inputs,
            n_transforms= n_transforms,
            n_neurons= n_neurons,
            n_conditional_inputs = n_conditional_inputs,
            n_blocks_per_transform = n_blocks_per_transform,
            batch_norm_between_transforms=True,
            dropout_probability=dp,
            linear_transform=linear_transform)
else: 
    raise ValueError('Flow not implemented')



best_model = copy.deepcopy(flow.state_dict())
best_val_loss = np.inf


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
device = device
flow.to(device)

#learning rate
lr = lr
print('LEARNING RATE = {}'.format(lr))

#Adam optimiser
optimiser_adam = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=0)

#Epochs
n_epochs = epochs
print('EPOCHS = {}'.format(n_epochs))

#Decay LR
decayRate = 0.999

if lr_scheduler == 'ExponentialLR':

    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser_adam, gamma=decayRate)
    
elif lr_scheduler == 'CyclicLR':

    my_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimiser_adam, base_lr=0.01, max_lr=0.03, step_size_up=75)   
    
elif lr_scheduler == 'CosineAnnealingLR':

    my_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser_adam, n_epochs, eta_min=0, last_epoch=- 1, verbose=False)   
    




# Loss and kl dictionaries
loss_dict = dict(train=[], val=[])
kl_dict =  dict(KL_vals1 = [], KL_vals2 = [], KL_vals3 = [], KL_vals4 = [], KL_vals5 = [], KL_vals6 = [], KL_vals7 = [], KL_vals8 = [])
                #KL_vals9 = [], KL_vals10 = [], KL_vals11 = [], KL_vals12 = [], KL_vals13 = [], KL_vals14 = [])

#save model hyperparameters
para = {'batch_size': batch_size,
          'n_epochs': n_epochs,
          'shuffle': True,
          'activation': None,
          'dropout': dp,
          'learning_rate': lr,
          'optimizer': 'Adam',
          'linear_transform':linear_transform,
          'n_neurons': int(n_neurons),
          'n_transforms': int(n_transforms),
          'n_blocks_per_transform': int(n_blocks_per_transform),
          'n_inputs': int(n_inputs),
          'n_conditional_inputs':int(n_conditional_inputs),
          'flow_type': flow_type,
          'log_it': log_it
          }

f = open(path+folder_name+'/hyperparameters.txt','w')
f.write(str(para))
f.close()

for j in range(n_epochs):
    
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
    
    #Save flow model at epoch
    if (j+1)%2000 == 0:
        torch.save(flow.state_dict(), path+folder_name+'/flows_epochs'+'/flow_epoch_{}.pt'.format(int(j+1)))
    
    if val_loss < best_val_loss:
        best_model = copy.deepcopy(flow.state_dict())
        best_val_loss = val_loss
    

    flow.eval()
    with torch.no_grad():
        #Latent space 
        conditionals = Y_scale_val.to(device)
        target_data = X_scale_val.to(device)
        latent_samples, _= flow.forward(target_data[:,:n_inputs], conditional=conditionals.reshape(-1,1))
        
        
    z_= latent_samples.cpu().detach().numpy()[0:10000]

    for i in range(n_inputs):
        _, kl_vals = KL_evaluate(z_[:,i])
        kl_dict[[*kl_dict][i]].append(kl_vals)
        
    with open(path+folder_name+'/loss_data.pickle', 'wb') as handle:
        pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path+folder_name+'/kl_data.pickle', 'wb') as handle:
        pickle.dump(kl_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    sys.stdout.write('\rEPOCH = {} || Training Value = {} || Validation Value = {}  '.format(j+1,round(loss_dict['train'][-1], 5), round(loss_dict['val'][-1], 5)))
    sys.stdout.flush()
    

flow.load_state_dict(best_model)

print()
print('Saving FLOW model in {}'.format(path+folder_name))
print()
#Save flow model 
torch.save(flow.state_dict(), path+folder_name+'/flow.pt')








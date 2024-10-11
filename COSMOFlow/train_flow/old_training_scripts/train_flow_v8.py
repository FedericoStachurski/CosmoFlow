from glasflow.flows import RealNVP, CouplingNSF
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
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
from cosmology_functions import utilities 
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
# ap.add_argument("-data", "--data_folder", required=False,
#    help="Name of the folder where training data is stored")
ap.add_argument("-batch", "--batch_size", required=False,
   help="batch size of the data to pass", default = 5000)
ap.add_argument("-n_cond", "--n_conditional", required=False,
   help="how many conditional variables", default = 1)
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
   help="learning_rate", default = 'No')
ap.add_argument("-linear_transform", "--linear_transform", required=False,
   help="type of transformation", default = None)
ap.add_argument("-drop_out", "--drop_out", required=False,
   help="drop_out", default = 0.0)
ap.add_argument("-log_it", "--log_it", required=False,
   help="logit the uniform distributions", default = 0)
ap.add_argument("-device", "--device", required=True,
   help="Which CUDA device [:0 , :1, :2]", default = 'cuda:1')
ap.add_argument("-xyz", "--xyz", required=True,
   help="convert to xyz coordinates, 1, else, 0", default = 1)
ap.add_argument("-Scaler", "--Scaler", required=True,
   help="choose how to scale the data: MinMax or Standard", default = 'MinMax')
ap.add_argument("-save_step", "--save_step", required=False,
   help="how many steps before it saves the flow model each time", default = 100)
ap.add_argument("-batches_data", "--batches_data", required=False,
   help="batcehs of the data", default = 1)
ap.add_argument("-VP", "--Volume_preserving", required=True,
   help="Choose if allow for volume perserving", default = False)


args = vars(ap.parse_args())
Name = str(args['Name_folder'])
# data = str(args['data_folder'])
n_conditional = int(args['n_conditional'])
flow_type = str(args['flow_type'])
batch_size = int(args['batch_size'])
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
xyz = int(args['xyz'])
Scaler = str(args['Scaler'])
save_steps = int(args['save_step'])
batches_data = int(args['batches_data'])
vp = args['Volume_preserving']

linear_transform = args['linear_transform']
if linear_transform is not None:
    linear_transform = str(args['linear_transform'])
elif  float(linear_transform) == 0:
    linear_transform = 0

print()
print('Name model = {}'.format(Name))
# print('data name = {}'.format(data))
print('batch = {}'.format(batch_size))
print('train size = {}'.format(train_size))
print('neurons = {}'.format(neurons))
print('layers = {}'.format(layers))
print('nblocks = {}'.format(nblock))
print('lr = {}'.format(lr))
print('lr_scheduler = {}'.format(lr_scheduler))
print('linear transform = {}'.format(linear_transform))
print('drop out = {}'.format(dp))
print('device = {}'.format(device))
print('xyz = {}'.format(xyz))
print()


#Save flow in folder 
folder_name = str(Name)
# path = 'trained_flows_and_curves/'
path = 'trained_flows_and_curves/'

#check if directory exists
if os.path.exists(path+folder_name):
    #if yes, delete directory
    shutil.rmtree(path+folder_name)


#Save model in folder
os.mkdir(path+folder_name)
os.mkdir(path+folder_name+'/flows_epochs')
os.chdir('..')

#run_O3_det_['H1', 'L1', 'V1']_name_NSBH_data_full_sky_128_catalog_True_band_K_batch_{}_N_250000_SNR_11_Nelect_2__Full_para_v1.csv
#read data to be used to train the flow 
def read_data(batch_of_data):
    path_name ="data_cosmoflow/galaxy_catalog/training_data_from_MLP/"
    data_name ="run_O3_det_['H1', 'L1', 'V1']_name_BBH_O3_events_all_para_catalog_True_band_K_batch_{}_N_250000_SNR_11_Nelect_4__Full_para_v1.csv".format(batch_of_data)
    print(data_name)
    GW_data = pd.read_csv(path_name+data_name,skipinitialspace=True)
                          # usecols=['snr', 'H0','gamma','kappa','zp', 'alpha', 'beta',
                          #                                                     'mmax', 'mmin', 'mu_g', 'sigma_g', 'lambda_peak', 'delta_m',
                          #                                                     'luminosity_distance', 'mass_1', 'mass_2', 'ra', 'dec','a_1', 'a_2', 'tilt_1',
                          #                                                     'tilt_2', 'theta_jn','phi_jl', 'phi_12', 'psi','geocent_time', 'app_mag', 'inx'])
    return GW_data

list_data = [] 
for i in range(batches_data): #how many batcehs of data to be used, i nthis case 1, (should be user input)
    list_data.append(read_data(i+1))

#concatenate batches together and perform sanity check of the data for repeats 
GW_data = pd.concat(list_data)
GW_data = GW_data.drop_duplicates(keep='first').sample(frac=1)
print()
print('Preview the data: SIZE = ({},{})'.format(len(GW_data), len(GW_data.columns)))
print()
print((GW_data.head()))  

#make dataframe structure correct, using only the columns needed
# data = GW_data[['luminosity_distance','mass_1', 'mass_2',
#                 'ra', 'dec', 'theta_jn', 'psi', 'geocent_time',
#                  'H0', 'gamma', 'kappa','zp', 'alpha', 'beta','mmax', 
#                 'mmin', 'mu_g', 'sigma_g', 'lambda_peak', 'delta_m']]

data = GW_data[['luminosity_distance','mass_1', 'mass_2',
                'ra', 'dec', 'theta_jn', 'psi', 'geocent_time',
                 'H0', 'gamma', 'k', 'zp', 'beta', 'alpha', 'mmax', 'mmin', 'mu_g', 'sigma_g', 'lambda_peak','delta_m']]
#, 'gamma', 'k', 'zp']]
#, 'gamma', 'mmax', 'mu_g']]



#convert geocentric time into siderael day seconds 
data['geocent_time'] = utilities.convert_gps_sday(data['geocent_time'])

#check if user wants to train data using cartesian or polar coordiantes 
if xyz == 1:
    #get polar coordinates 
    coordinates= data[['luminosity_distance', 'ra', 'dec']]
    dl = np.array(coordinates.luminosity_distance)
    ra = np.array(coordinates.ra)
    dec = np.array(coordinates.dec)

    x,y,z = cosmology.spherical_to_cart(dl, ra, dec)

    #add cartesian coordiantes to dataframe 
    data.loc[:,'xcoord'] = x
    data.loc[:,'ycoord'] = y
    data.loc[:,'zcoord'] = z
    data = data[['xcoord', 'ycoord', 'zcoord', 'mass_1', 'mass_2', 'H0', 'Om0', 'gamma', 'kappa','zp', 'alpha', 'beta','mmax', 
                'mmin', 'mu_g', 'sigma_g', 'lambda_peak', 'delta_m']]
else: 
    # data = data[['luminosity_distance', 'ra', 'dec', 'mass_1', 'mass_2', 'H0', 'gamma', 'kappa','zp', 'alpha', 'beta','mmax', 
    #             'mmin', 'mu_g', 'sigma_g', 'lambda_peak', 'delta_m']]
    # data = data[['luminosity_distance', 'ra', 'dec', 'mass_1', 'mass_2', 'H0', 'gamma']]
    data = data[['luminosity_distance', 'ra', 'dec', 'mass_1', 'mass_2', 'H0',
                 'gamma', 'k', 'zp', 'beta','alpha', 'mmax', 'mmin', 'mu_g','sigma_g', 'lambda_peak','delta_m' ]]
                 #, 'gamma', 'mmax', 'mu_g']]

#print some data to check beofre it starts training 
print(data.head(10))

#scale input data using MinMax or Standard scaler from sklearn, get target scaler, x, conditional scaler, y, and the scaled data 
scaler_x, scaler_y, scaled_data = utilities.scale_data(data, Scaler, n_conditionals = n_conditional)

#if log_it = 1, perfomr log it transofmration on scaeld data, only the columns where the boundaries are more prominent 
if log_it == 1:
    scaled_data = it_data(scaled_data)
    scaled_data = scaled_data[np.isfinite(scaled_data).all(1)]

    
#split the data in training and validation data sets 
x_train, x_val = train_test_split(scaled_data, test_size = float(1.0 - train_size), train_size = float(train_size))

train_tensor = torch.from_numpy(np.asarray(x_train).astype('float32')) #get training data tensor (torch) 
val_tensor = torch.from_numpy(np.asarray(x_val).astype('float32')) #get validationsor (torch) 


X_scale_train = train_tensor[:,:-n_conditional] #get target data for training 
X_scale_val = val_tensor[:,:-n_conditional] #get target data for validation

if n_conditional == 1:
    Y_scale_train = train_tensor[:,-n_conditional] #get conditional data for training 
    Y_scale_val = val_tensor[:,-n_conditional] #get conditional data for validation 
else: 
    Y_scale_train = train_tensor[:,-n_conditional:] #get conditional data for training 
    Y_scale_val = val_tensor[:,-n_conditional:] #get conditional data for validation 

train_dataset = torch.utils.data.TensorDataset(X_scale_train ,Y_scale_train) #put both target and conditional data sets into one torch tensor 
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True) #initialize data loader 

val_dataset = torch.utils.data.TensorDataset(X_scale_val , Y_scale_val)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                           shuffle=False) #initialize dataloader for the validation data set 


conditional_val = scaler_y.inverse_transform(np.array(Y_scale_val).reshape(-1, n_conditional) ) #unscale validation data set 
target_val = scaler_x.inverse_transform(X_scale_val)


# Save data scalers 
print()
print('Saving Scalers X and Y')
print()
print(os.getcwd())
scalerfileX = 'train_flow/'+path+folder_name+'/scaler_x.sav'
scalerfileY = 'train_flow/'+path+folder_name+'/scaler_y.sav'
pickle.dump(scaler_x, open(scalerfileX, 'wb'))
pickle.dump(scaler_y, open(scalerfileY, 'wb'))


# print sizes of data 
print()
print('Train loader size = {}'.format(len(train_loader)))
print('Batch size = {}'.format(batch_size))
print('Train/Validation = {}% / {}%'.format(train_size * 100, round(1- train_size, 2)*100))
print()


# Define inputs for flow
n_inputs = len(data.columns) - n_conditional
n_conditional_inputs = n_conditional 
n_neurons = neurons
n_transforms = layers
n_blocks_per_transform = nblock


print()
print('n_inputs = {} ; n_conditional_inputs = {}'.format(n_inputs, n_conditional_inputs))
print('NEURONS = {}; LAYERS = {}; BLOCKS = {}'.format(n_neurons, n_transforms, n_blocks_per_transform))
print()


#Define which type of Flow 
if flow_type == 'RealNVP':
    flow = RealNVP(n_inputs= n_inputs,
            n_transforms= n_transforms,
            n_neurons= n_neurons,
            n_conditional_inputs = n_conditional_inputs,
            n_blocks_per_transform = n_blocks_per_transform,
            batch_norm_between_transforms=True,
            dropout_probability=dp,
            linear_transform=linear_transform,
            volume_preserving = vp)

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

#Decay LR
decayRate = 0.999
if lr_scheduler != 'No':
    if lr_scheduler == 'ExponentialLR':

        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser_adam, gamma=decayRate)

    elif lr_scheduler == 'CyclicLR':

        my_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimiser_adam, base_lr=0.01, max_lr=0.03, step_size_up=75)   

    elif lr_scheduler == 'CosineAnnealingLR':

        my_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser_adam, n_epochs, eta_min=0, last_epoch=- 1, verbose=False)   





# Loss and kl dictionaries
loss_dict = dict(train=[], val=[])
kl_dict =  dict(KL_vals1 = [], KL_vals2 = [], KL_vals3 = [], KL_vals4 = [], KL_vals5 = [], KL_vals6 = [], KL_vals7 = [], KL_vals8 = [])


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
          'log_it': log_it,
          'xyz' : xyz,
          'scaler': Scaler,
          'lr_scheduler': lr_scheduler,
          'volume_preserving': vp
            
          }

f = open('train_flow/'+path+folder_name+'/hyperparameters.txt','w')
f.write(str(para))
f.close()


for j in range(n_epochs):
    
    optimiser = optimiser_adam

    flow.to(device)

    #Train
    train_loss = 0
    for batch in train_loader:
        #get target and target and conditional training data from batch 
        target_train, conditionals_train = batch 
        flow.train() #set flow in train mode 
        optimiser.zero_grad() # set derivatives to zero 
        
        # print( scaler_y.inverse_transform(conditionals_train.reshape(-1,n_conditional)))
        
        loss = -flow.log_prob(target_train.to(device)[:,:n_inputs], conditional=conditionals_train.reshape(-1,n_conditional).to(device)).cpu().mean() #compute loss
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
            target_val, condtionals_val = batch #get validation data fro each batch 
            loss = - flow.log_prob(target_val.to(device)[:,:n_inputs],conditional=condtionals_val.reshape(-1,n_conditional).to(device)).cpu().mean() #perform loss function for validatio ndata set 
            val_loss += loss.item() #add validation values from each batch 
    val_loss /= len(val_loader) #get mean validation value 
    loss_dict['val'].append(val_loss) #append validation value 
    
    #Save flow model at epoch
    if (j+1)%save_steps == 0: #every 100 steps, save the flow state 
        torch.save(flow.state_dict(), 'train_flow/'+path+folder_name+'/flows_epochs'+'/flow_epoch_{}.pt'.format(int(j+1)))
    
    if val_loss < best_val_loss: #if validation loss is better than the previous step, save new model 
        best_model = copy.deepcopy(flow.state_dict())
        best_val_loss = val_loss
    
    with torch.no_grad():
        #Latent space 
        conditionals = Y_scale_val.to(device)
        target_data = X_scale_val.to(device)
        latent_samples, _= flow.forward(target_data[:,:n_inputs], conditional=conditionals.reshape(-1,n_conditional)) #get samples from latent sapce of flow
        
        
    z_= latent_samples.cpu().detach().numpy()[0:10000] #10000 samples from latent space 

    for i in range(n_inputs):
        _, kl_vals = utilities.KL_evaluate_gaussian(z_[:,i], gaussian, g) #evaluate KL between one dimension of the latent space and a gaussian 
        kl_dict[[*kl_dict][i]].append(kl_vals) #save KL valu fro each entry 

    latent_samples = dict(z_samples=z_)
    
    #save loss and kl values data in pickle files 
    with open('train_flow/'+path+folder_name+'/loss_data.pickle', 'wb') as handle:
        pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('train_flow/'+path+folder_name+'/kl_data.pickle', 'wb') as handle:
        pickle.dump(kl_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('train_flow/'+path+folder_name+'/latent_data.pickle', 'wb') as handle:
        pickle.dump(latent_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    sys.stdout.write('\rEPOCH = {} || Training Value = {} || Validation Value = {}  '.format(j+1,round(loss_dict['train'][-1], 5), round(loss_dict['val'][-1], 5)))
    sys.stdout.flush()
    
#save flow model 
flow.load_state_dict(best_model)

print()
print('Saving FLOW model in {}'.format(path+folder_name))
print()
#Save flow model 
torch.save(flow.state_dict(), 'train_flow/'+path+folder_name+'/flow.pt')








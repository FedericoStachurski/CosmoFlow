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
# python3 train_flow_v3.py -Name Flow_Glade_mth_v1-batch 25000 -train_size 0.8 -flow_type RealNVP -epochs 10 -neurons 128 -layers 6 -nblock 4 -lr 0.001



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


linear_transform = args['linear_transform']
if linear_transform is not None:
    linear_transform = str(args['linear_transform'])

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
    data_name = "name_training_data_catalog_True_band_K_batch_{}_N_250000_SNR_11_Nelect_5__Full_para_v1.csv".format(batch)
    GW_data = pd.read_csv(path_name+data_name,skipinitialspace=True, usecols=['H0', 'dl','m1', 'm2','a1', 'a2', 'tilt1', 
                                                                              'tilt2', 'RA', 'dec', 'theta_jn', 'phi_jl', 
                                                                             'phi_12', 'polarization', 
                                                                              'geo_time',  'app_mag', 'z'])
    return GW_data

list_data = [] 
for i in range(5):
    list_data.append(read_data(i+1))


GW_data = pd.concat(list_data)
print()
print('Preview the data: SIZE = ({},{})'.format(len(GW_data), len(GW_data.columns)))
print()
print((GW_data.head()))   


data = GW_data[['dl','m1', 'm2','a1', 'a2', 'tilt1', 'tilt2', 
                'RA', 'dec', 'theta_jn', 'phi_jl', 'phi_12', 'polarization', 'geo_time','app_mag', 'z', 'H0']]

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



data = data[['xcoord', 'ycoord', 'zcoord', 'm1', 'm2','a1', 'a2', 
             'tilt1', 'tilt2', 'theta_jn', 'phi_jl','phi_12', 'polarization', 'geo_time','app_mag', 'z',  'H0']]

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
x_train, x_val = train_test_split(scaled_data, test_size=train_size)

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
n_inputs = 16
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
device = 'cuda:1'
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

for j in range(n_epochs):
    
    optimiser = optimiser_adam

  #  if j == int(n_epochs-100): 
   #     #SGD optmiser 
 #       optimiser_sgd = torch.optim.SGD(flow.parameters(), lr=lr, weight_decay=0)
#        optimiser = optimiser_sgd

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
    if (j+1)%100 == 0:
        torch.save(flow.state_dict(), path+folder_name+'/flows_epochs'+'/flow_epoch_{}.pt'.format(int(j+1)))
    
    if val_loss < best_val_loss:
        best_model = copy.deepcopy(flow.state_dict())
        best_val_loss = val_loss
    

    flow.eval()
    with torch.no_grad():
        conditionals = Y_scale_val.to(device)
        target_data = X_scale_val.to(device)
        latent_samples, _= flow.forward(target_data[:,:n_inputs], conditional=conditionals.reshape(-1,1))
    z_= latent_samples.cpu().detach().numpy()[0:10000]

    #KDEsdensity.pdf(g)
    kde_points1, kl_val_1 = KL_evaluate(z_[:,0])
    kde_points2, kl_val_2 = KL_evaluate(z_[:,1])
    kde_points3, kl_val_3 = KL_evaluate(z_[:,2])
    kde_points4, kl_val_4 = KL_evaluate(z_[:,3])
    kde_points5, kl_val_5 = KL_evaluate(z_[:,4])
    kde_points6, kl_val_6 = KL_evaluate(z_[:,5])
    kde_points7, kl_val_7 = KL_evaluate(z_[:,6])
    kde_points8, kl_val_8 = KL_evaluate(z_[:,7])
    kde_points9, kl_val_9 = KL_evaluate(z_[:,8])
    kde_points10, kl_val_10 = KL_evaluate(z_[:,9])
    kde_points11, kl_val_11 = KL_evaluate(z_[:,10])
    kde_points12, kl_val_12 = KL_evaluate(z_[:,11])
    kde_points13, kl_val_13 = KL_evaluate(z_[:,12])
    kde_points14, kl_val_14 = KL_evaluate(z_[:,13])
    
    

    KL_vals1.append(kl_val_1)
    KL_vals2.append(kl_val_2)
    KL_vals3.append(kl_val_3)
    KL_vals4.append(kl_val_4)
    KL_vals5.append(kl_val_5)
    KL_vals6.append(kl_val_6)
    KL_vals7.append(kl_val_7)
    KL_vals8.append(kl_val_8)
    KL_vals9.append(kl_val_9)
    KL_vals10.append(kl_val_10)
    KL_vals11.append(kl_val_11)
    KL_vals12.append(kl_val_12)
    KL_vals13.append(kl_val_13)
    KL_vals14.append(kl_val_14)

    
 
    #Real time loss plotting
    #Define figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25,15))

    #ax1.set_title('lr = ' + str(lr1))
    ax1.plot(np.linspace(1,j+1, len(loss_dict['train'])), loss_dict['train'],'k', label='Train', linewidth = 3)
    ax1.plot(np.linspace(1,j+1, len(loss_dict['train'])), loss_dict['val'],'r', label='Validation', alpha=0.4, linewidth = 3)
    ax1.set_ylabel('loss', fontsize = 20)
    ax1.set_xlabel('Epochs', fontsize = 20)
    ax1.set_xscale('log')
    ax1.set_ylim([np.min(loss_dict['train'])-0.5,np.max(loss_dict['train'])])
    ax1.set_xlim([1,n_epochs])
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    ax1.grid(True) 
    ax1.legend(fontsize = 20)

    #Real time latent space plotting        

    ax2.set_ylim([0,0.5])
    ax2.set_xlim([-5,5])
    ax2.plot(g, kde_points1, linewidth=3,alpha = 0.6, label = r'$z_{1}$')
    ax2.plot(g, kde_points2, linewidth=3,alpha = 0.6, label = r'$z_{2}$')
    ax2.plot(g, kde_points3, linewidth=3,alpha = 0.6, label = r'$z_{3}$')
    ax2.plot(g, kde_points4, linewidth=3,alpha = 0.6, label = r'$z_{4}$')
    ax2.plot(g, kde_points5, linewidth=3,alpha = 0.6, label = r'$z_{5}$')
    ax2.plot(g, kde_points6, linewidth=3,alpha = 0.6, label = r'$z_{6}$')
    ax2.plot(g, kde_points7, linewidth=3,alpha = 0.6, label = r'$z_{7}$')
    ax2.plot(g, kde_points8, linewidth=3,alpha = 0.6, label = r'$z_{8}$')
    ax2.plot(g, kde_points9, linewidth=3,alpha = 0.6, label = r'$z_{9}$')
    ax2.plot(g, kde_points10, linewidth=3,alpha = 0.6, label = r'$z_{10}$')
    ax2.plot(g, kde_points11, linewidth=3,alpha = 0.6, label = r'$z_{11}$')
    ax2.plot(g, kde_points12, linewidth=3,alpha = 0.6, label = r'$z_{12}$')
    ax2.plot(g, kde_points13, linewidth=3,alpha = 0.6, label = r'$z_{13}$')
    ax2.plot(g, kde_points14, linewidth=3,alpha = 0.6, label = r'$z_{14}$')
#     ax2.plot(g, kde_points6, linewidth=3, label = r'$z_{5}$')
    ax2.plot(g, gaussian,linewidth=5,c='k',label=r'$\mathcal{N}(0;1)$')

    ax2.legend(fontsize = 15)
    ax2.grid(True)
    ax2.set_ylabel('$p(z)$',fontsize = 20)
    ax2.set_xlabel('$z$',fontsize = 20)
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)

    #Real time JS div between gaussian and latent 
    ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), KL_vals1,linewidth=3,alpha = 0.6, label = r'$z_{l}$')
    ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), KL_vals2,linewidth=3,alpha = 0.6,  label = r'$z_{2}$')
    ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), KL_vals3,linewidth=3,alpha = 0.6,  label = r'$z_{3}$')
    ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), KL_vals4,linewidth=3,alpha = 0.6,  label = r'$z_{4}$')
    ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), KL_vals5,linewidth=3,alpha = 0.6,  label = r'$z_{5}$')
    ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), KL_vals6,linewidth=3,alpha = 0.6, label = r'$z_{6}$')
    ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), KL_vals7,linewidth=3,alpha = 0.6,  label = r'$z_{7}$')
    ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), KL_vals8,linewidth=3,alpha = 0.6,  label = r'$z_{8}$')
    ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), KL_vals9,linewidth=3,alpha = 0.6,  label = r'$z_{9}$')
    ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), KL_vals10,linewidth=3,alpha = 0.6,  label = r'$z_{10}$')
    ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), KL_vals11,linewidth=3,alpha = 0.6,  label = r'$z_{11}$')
    ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), KL_vals12,linewidth=3,alpha = 0.6,  label = r'$z_{12}$')
    ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), KL_vals13,linewidth=3,alpha = 0.6,  label = r'$z_{13}$')
    ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), KL_vals14,linewidth=3,alpha = 0.6,  label = r'$z_{14}$')
#     ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), JS_vals6,linewidth=3,alpha = 0.6,  label = r'$z_{5}$')

    ax3.set_ylabel('KLDiv', fontsize = 20)
    ax3.set_xlabel(r'Epochs', fontsize = 20)
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.set_xlim([1,n_epochs])
    ax3.grid(True)
    ax3.xaxis.set_tick_params(labelsize=20)
    ax3.yaxis.set_tick_params(labelsize=20)
    ax3.legend(fontsize = 15)
    fig.tight_layout()
    fig.savefig(path+folder_name+'/training.png', dpi = 50 )
        
    plt.close('all')   # Clear figure




    sys.stdout.write('\rEPOCH = {} || Training Value = {} || Validation Value = {}  '.format(j+1,round(loss_dict['train'][-1], 5), round(loss_dict['val'][-1], 5)))
    sys.stdout.flush()
    

flow.load_state_dict(best_model)

print()
print('Saving FLOW model in {}'.format(path+folder_name))
print()
#Save flow model 
torch.save(flow.state_dict(), path+folder_name+'/flow.pt')

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






# ##################################### TESTING #####################################

# ###### TEST 1: PP-Plot ######


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
#                           'm1', 'm2','a1',
#                           'a2', 'tilt1', 'tilt2',
#                           'theta_jn', 'phi_jl', 'phi_12',
#                           'polarization', 'geo_time']] , plot_datapoints=False, smooth = True, fig = c1, levels = (0.5, 0.9), plot_density=True,labels=[r'$x[Mpc]$',r'$y[Mpc]$',r'$z[Mpc]$', r'$m_{1,z}$', r'$m_{2,z}$',r'$loga_{1}$', r'$loga_{2}$', r'$tilt_{1}$', r'$tilt_{2}$', r'$\theta_{JN}$', r'$log\phi_{JL}$',  r'$log\phi_{12}$',r'$log\psi$', r'$logt_{geo}$'], hist_kwargs = {'density' : 1})



# plt.savefig(path+folder_name+'/flow_resample.png', dpi = 100)




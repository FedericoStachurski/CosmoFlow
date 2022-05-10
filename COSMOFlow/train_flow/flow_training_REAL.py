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
bilby.core.utils.log.setup_logger(log_level=0)

import argparse


#pass arguments 
# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-Name", "--Name_folder", required=True,
   help="Name of the folder to save the FLOW model")
ap.add_argument("-data", "--data_folder", required=True,
   help="Name of the folder where training data is stored")
ap.add_argument("-batch", "--batch_size", required=False,
   help="batch size of the data to pass", default = 5000)
ap.add_argument("-train_size", "--train_size", required=False,
   help="size of trainign data percentage", default = 0.75)
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





args = vars(ap.parse_args())
Name = str(args['Name_folder'])
data = str(args['data_folder'])
batch = int(args['batch_size'])
train_size = float(args['train_size'])
neurons = int(args['neurons'])
layers = int((args['layers']))
nblock = int(args['nblock'])
lr = float(args['learning_rate'])
epochs = int(args['epochs'])

print()
print('Name model = {}'.format(Name))
print('data name = {}'.format(data))
print('batch = {}'.format(batch))
print('train size = {}'.format(train_size))
print('neurons = {}'.format(neurons))
print('layers = {}'.format(layers))
print('nblocks = {}'.format(nblock))
print('lr = {}'.format(lr))
print()






folder_name = str(Name)
path = 'trained_flows_and_curves/'

def kl_div(p, q):
    "Compute KL-divergence between p and q prob arrays"
    return np.sum(np.where(p != 0, p*np.log(p/q), 0))

def js_div(p,q):
    "Compute the JS-divergence between p and q prob arrays"
    return 0.5*(kl_div(p,q) + kl_div(q,p))


#check if directory exists
if os.path.exists(path+folder_name):
    #if yes, delete directory
    shutil.rmtree(path+folder_name)

#Save model in folder
os.mkdir(path+folder_name)

os.chdir('..')


def read_data(batch):
    path_name ="data_gwcosmo/galaxy_catalog/training_data/"
    data_name = "glade_test_data_1000_batch_{}.csv".format( batch)
    GW_data = pd.read_csv(path_name+data_name,skipinitialspace=True, usecols=['H0', 'dl','m1', 'm2', 'ra', 'dec', 'm_B'])
    return GW_data

list_data = [] 
for i in range(10):
    list_data.append(read_data(i+1))




GW_data = pd.concat(list_data)
print()
print('Preview the data: SIZE = ({},{})'.format(len(GW_data), len(GW_data.columns)))
print()
print((GW_data.head()))   



data = GW_data[['dl','m1', 'm2', 'ra', 'dec','H0']]

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

batch_size=500

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




flow = RealNVP(n_inputs= n_inputs,
        n_transforms= n_transforms,
        n_neurons= n_neurons,
        n_conditional_inputs = n_conditional_inputs,
        n_blocks_per_transform = n_blocks_per_transform,
        batch_norm_between_transforms=True,
        dropout_probability=0.0,
        linear_transform='permutation')

best_model = copy.deepcopy(flow.state_dict())
best_val_loss = np.inf


#Standard Gaussian
g = np.linspace(-5, 5, 1000)
gaussian = norm.pdf(g)

def JS_evaluate(samples):
    density = gaussian_kde(samples)
    kde_points = density.pdf(g)
    return np.array(kde_points), js_div(kde_points, gaussian)

##################################### TRAINING #####################################
#Define device GPU or CPU
device = 'cuda:2'
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
#my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimiser, gamma=decayRate)
#my_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimiser, base_lr=0.01, max_lr=0.03, step_size_up=75)
#my_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, n_epochs, eta_min=0, last_epoch=- 1, verbose=False)


# Loss
loss_dict = dict(train=[], val=[])


#stor KL values at each epoch 
JS_vals1 = []
JS_vals2 = []
JS_vals3 = []
JS_vals4 = []
JS_vals5 = []
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
    #my_lr_scheduler.step()

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
    
    if val_loss < best_val_loss:
        best_model = copy.deepcopy(flow.state_dict())
        best_val_loss = val_loss
    

    flow.eval()
    with torch.no_grad():
        conditionals = Y_scale_val.to(device)
        target_data = X_scale_val.to(device)
        latent_samples, _= flow.forward(target_data[:,:n_inputs], conditional=conditionals.reshape(-1,1))
    z_= latent_samples.cpu().detach().numpy()[0:5000]

    #KDEsdensity.pdf(g)
    kde_points1, js_val_1 = JS_evaluate(z_[:,0])
    kde_points2, js_val_2 = JS_evaluate(z_[:,1])
    kde_points3, js_val_3 = JS_evaluate(z_[:,2])
    kde_points4, js_val_4 = JS_evaluate(z_[:,3])
    kde_points5, js_val_5 = JS_evaluate(z_[:,4])
#     kde_points6, js_val_6 = JS_evaluate(z_[:,5])
    
    JS_vals1.append(js_val_1)
    JS_vals2.append(js_val_2)
    JS_vals3.append(js_val_3)
    JS_vals4.append(js_val_4)
    JS_vals5.append(js_val_5)
#     JS_vals6.append(js_val_6)

    
 
    #Real time loss plotting
    #Define figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25,15))

    #ax1.set_title('lr = ' + str(lr1))
    ax1.plot(np.linspace(1,j+1, len(loss_dict['train'])), loss_dict['train'],'k', label='Train')
    ax1.plot(np.linspace(1,j+1, len(loss_dict['train'])), loss_dict['val'],'r', label='Validation', alpha=0.5)
    ax1.set_ylabel('loss', fontsize = 20)
    ax1.set_xlabel('Epochs', fontsize = 20)
    ax1.set_xscale('log')
    #ax1.set_ylim([-2.0,0.1])
    ax1.set_xlim([1,n_epochs])
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    ax1.grid(True) 
    ax1.legend(fontsize = 20)

    #Real time latent space plotting        

    ax2.set_ylim([0,0.5])
    ax2.set_xlim([-5,5])
    ax2.plot(g, kde_points1, linewidth=3,alpha = 0.6, label = r'$z_{dl}$')
    ax2.plot(g, kde_points2, linewidth=3,alpha = 0.6, label = r'$z_{m1}$')
    ax2.plot(g, kde_points3, linewidth=3,alpha = 0.6, label = r'$z_{m2}$')
    ax2.plot(g, kde_points4, linewidth=3,alpha = 0.6, label = r'$z_{RA}$')
    ax2.plot(g, kde_points5, linewidth=3,alpha = 0.6, label = r'$z_{dec}$')
#     ax2.plot(g, kde_points6, linewidth=3, label = r'$z_{5}$')
    ax2.plot(g, gaussian,linewidth=5,c='k',label=r'$\mathcal{N}(0;1)$')

    ax2.legend(fontsize = 15)
    ax2.grid(True)
    ax2.set_ylabel('$p(z)$',fontsize = 20)
    ax2.set_xlabel('$z$',fontsize = 20)
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)

    #Real time JS div between gaussian and latent 
    ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), JS_vals1,linewidth=3,alpha = 0.6, label = r'$z_{dl}$')
    ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), JS_vals2,linewidth=3,alpha = 0.6,  label = r'$z_{m1}$')
    ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), JS_vals3,linewidth=3,alpha = 0.6,  label = r'$z_{m2}$')
    ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), JS_vals4,linewidth=3,alpha = 0.6,  label = r'$z_{RA}$')
    ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), JS_vals5,linewidth=3,alpha = 0.6,  label = r'$z_{dec}$')
#     ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), JS_vals6,linewidth=3,alpha = 0.6,  label = r'$z_{5}$')
    ax3.set_ylabel('JS Div', fontsize = 20)
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
          'dropout': 0.0,
          'learning_rate': lr,
          'optimizer': 'Adam',
          'linear_transform':'permutation',
          'n_neurons': int(n_neurons),
          'n_transforms': int(n_transforms),
          'n_blocks_per_transform': int(n_blocks_per_transform),
          'n_inputs': int(n_inputs),
          'n_conditional_inputs':int(n_conditional_inputs)
          }

f = open(path+folder_name+'/hyperparameters.txt','w')
f.write(str(para))
f.close()







##################################### TESTING #####################################


print()
print('Making Probability-Probability plot with Validation data')
print()

def Flow_samples(conditional, n):
    
    Y_H0_conditional = scaler_y.transform(conditional.reshape(-1,1))

    
    conditional = np.array(Y_H0_conditional)
    data = np.array(conditional)
    data_scaled = torch.from_numpy(data.astype('float32'))
    
    flow.eval()
    flow.to('cpu')
    

    with torch.no_grad():
        samples = flow.sample(n, conditional=data_scaled.to('cpu'))
        samples= scaler_x.inverse_transform(samples.to('cpu'))
    return samples 



np.random.seed(1234)
Nresults =200
Nruns = 1
labels = ['dl', 'm1', 'm2', 'RA', 'dec']
priors = {}
for jj in range(5):
    priors.update({f"{labels[jj]}": Uniform(0, 1, f"{labels[jj]}")})




for x in range(Nruns):
    results = []
    for ii in tqdm(range(Nresults)):
        posterior = dict()
        injections = dict()
        i = 0 
        for key, prior in priors.items():

            inx = np.random.randint(len(Y_scale_val))  
            truths=  scaler_x.inverse_transform(X_scale_val[inx,:].reshape(1,-1))[0]
            conditional_sample = scaler_y.inverse_transform(Y_scale_val[inx].reshape(1,-1))[0]
            conditional_sample = conditional_sample *np.ones(5000)
            samples = Flow_samples(conditional_sample, 5000)
            posterior[key] = samples[:,i] 
            injections[key] = truths[i].astype('float32').item()
            i += 1

        posterior = pd.DataFrame(dict(posterior))
        result = bilby.result.Result(
            label="test",
            injection_parameters=injections,
            posterior=posterior,
            search_parameter_keys=injections.keys(),
        priors = priors )
        results.append(result)

    fig = bilby.result.make_pp_plot(results, filename=path+folder_name+'/PP',
                              confidence_interval=(0.68, 0.90, 0.99, 0.9999))






#TEST 2: Resample target data 

N = 50000

combined_samples = []
total_H0_samples = []
while True: 
    
    H0_samples = np.random.uniform(20,120,N)

    samples = Flow_samples(H0_samples, N)


    #Condtions 
    
#     z = cosmology.fast_dl_to_z_v2(samples[:,0], H0_samples)
    
#     samples = np.concatenate([samples, H0_samples, z], axis=1)
    samples = samples[np.where(samples[:,0] > 0)[0], :]
    samples = samples[np.where(samples[:,1] > 0)[0], :]
    samples = samples[np.where(samples[:,2] > 0)[0], :]
    samples = samples[np.where((samples[:,3] > 0) & (samples[:,3] <= 2*np.pi))[0], :]
    samples = samples[np.where((samples[:,4] > -np.pi/2) & (samples[:,4] <= np.pi/2))[0], :]

#     m1 = (1/(1+a)) * samples[:,1]
#     m2 = (1/(1+a)) * samples[:,2]
#     sumM = m1 + m2 
#     indicies = np.where(sumM <= 100)[0]
#     samples = samples[indicies,:]

    combined_samples = combined_samples + list(samples)

    
    if len(np.array(combined_samples)) >= N:
        combined_samples = np.array(combined_samples)[:N,:]
        break




c1 = corner.corner(combined_samples, smooth = True, color = 'red', hist_kwargs = {'density' : 1})
fig = corner.corner(data[['dl', 'm1', 'm2', 'ra', 'dec']], smooth = True, fig = c1, plot_density=True,labels=[r'$D_{L}$', r'$m_{1,z}$', r'$m_{2,z}$', r'RA', r'$\delta$'], hist_kwargs = {'density' : 1})

plt.savefig(path+folder_name+'/flow_resample.png', dpi = 100)




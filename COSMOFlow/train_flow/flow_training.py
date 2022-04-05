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

folder_name = 'flow_v4_sky'
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

#%% initial parameters 
rho_th = 8                 # the SNR threshold
mth = 23.0                   #the magnitude threshold
NH = 200
short_Hvec = np.linspace(20,120,20)
Hvec = np.linspace(20,120,NH)  # a vector of H0 values
zmax =1.0  # max redshift
Mmin = 15  #chirp mass Min 
Mmax = 100 #chirp mass Max


def read_data(batch):
    path_name ="/data/wiay/federico/PhD/gwcosmoFlow_v3/data_gwcosmo/events_mock_cat_zmax_1_mth_23_sky/training_data/"
    data_name = "data_mth_{}_SNRth_{}_5000_batch_{}.csv".format(int(mth), rho_th, batch)
    GW_data = pd.read_csv(path_name+data_name,skipinitialspace=True, usecols=['H0', 'SNR', 'Mz','RA','dec', 'z'])
    return GW_data

list_data = [] 
for i in range(20):
    list_data.append(read_data(i+1))



GW_data = pd.concat(list_data)
print()
print('Preview the data: SIZE = ({},{})'.format(len(GW_data), len(GW_data.columns)))
print()
print((GW_data.head()))   


#Prepare training data
RN = np. random.normal(0,1, size= len(GW_data))
data = GW_data[['H0', 'SNR', 'Mz','RA','dec', 'z']]
data.insert(1, 'RN',RN, True)
data = data[['H0', 'RN','SNR', 'Mz','RA','dec', 'z']]

def scale_data(data_to_scale):
    target = data_to_scale[data_to_scale.columns[0:2]]
    conditioners = data_to_scale[data_to_scale.columns[2:]]
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaled_target = scaler_x.fit_transform(target) 
    scaled_conditioners = scaler_y.fit_transform(conditioners)  
    scaled_data = np.hstack((scaled_target, scaled_conditioners))
    scaled_data = pd.DataFrame(scaled_data, index=data_to_scale.index, columns=data_to_scale.columns)
    return scaler_x, scaler_y, scaled_data
    
scaler_x, scaler_y, scaled_data = scale_data(data)
x_train, x_val = train_test_split(scaled_data, test_size=0.25)

batch_size=25000

train_tensor = torch.from_numpy(np.asarray(x_train).astype('float32'))
val_tensor = torch.from_numpy(np.asarray(x_val).astype('float32'))


X_scale_train = train_tensor[:,:2]
Y_scale_train = train_tensor[:,2:]
X_scale_val = val_tensor[:,:2]
Y_scale_val = val_tensor[:,2:]

train_dataset = torch.utils.data.TensorDataset(X_scale_train ,Y_scale_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True)

val_dataset = torch.utils.data.TensorDataset(X_scale_val , Y_scale_val)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                        shuffle=False)


rho_Mz_val = scaler_y.inverse_transform(Y_scale_val)
H0_RN_val = scaler_x.inverse_transform(X_scale_val)



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
print('Train/Validation = {}% / {}%'.format(75,25))
print()


# Define Flow
n_inputs = 2
n_conditional_inputs = 4
n_neurons = 36
n_transforms = 3
n_blocks_per_transform = 2


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
        linear_transform=None)

best_model = copy.deepcopy(flow.state_dict())
best_val_loss = np.inf



##################################### TRAINING #####################################

#Define device GPU or CPU
device = 'cuda:2'
flow.to(device)

#learning rate
lr = 0.001
print('LEARNING RATE = {}'.format(lr))

#Adam optimiser
optimiser_adam = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=0)



#Epochs
n_epochs = 20
print('EPOCHS = {}'.format(n_epochs))

#Decay LR
decayRate = 0.999
#my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimiser, gamma=decayRate)
#my_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimiser, base_lr=0.01, max_lr=0.03, step_size_up=75)
#my_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, n_epochs, eta_min=0, last_epoch=- 1, verbose=False)


# Loss
loss_dict = dict(train=[], val=[])

#Standard Gaussian
g = np.linspace(-5, 5, 1000)
gaussian = norm.pdf(g)



#stor KL values at each epoch 
JS_vals1 = []
JS_vals2 = []

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
        
        target_train, condtionals_train = batch 
        flow.train()
        optimiser.zero_grad()
        loss = -flow.log_prob(target_train.to(device)[:,:2], conditional=condtionals_train[:,:n_conditional_inputs].to(device)).cpu().mean()
        loss.backward()
        
        #optimiser step
        
        optimiser.step()


        train_loss += loss.item()

    train_loss /= len(train_loader)
    loss_dict['train'].append(train_loss)
    lr_upd = optimiser.param_groups[0]["lr"]
    #my_lr_scheduler.step()

    #Validate
    with torch.no_grad():
        val_loss = 0
        for batch in val_loader: 
            target_val, condtionals_val = batch 
            flow.eval()

            loss = - flow.log_prob(target_val.to(device)[:,:2],
                                   conditional=condtionals_val[:,:n_conditional_inputs].to(device)).cpu().mean()
            val_loss += loss.item()       
    val_loss /= len(val_loader)
    loss_dict['val'].append(val_loss)
    
    if val_loss < best_val_loss:
        best_model = copy.deepcopy(flow.state_dict())
        best_val_loss = val_loss
    

    flow.eval()
    with torch.no_grad():
        conditionals = Y_scale_val.to(device)[:,:n_conditional_inputs]
        target_data = X_scale_val.to(device)
        latent_samples, _= flow.forward(target_data[:,:2], conditional=conditionals)
    z_= latent_samples.cpu().detach().numpy()[0:5000]

    #KDEsdensity.pdf(g)
    density = gaussian_kde(z_[:,0])
    kde_points1 = density.pdf(g)
    JS_vals1.append(js_div(kde_points1, gaussian))

    density = gaussian_kde(z_[:,1])
    kde_points2 = density.pdf(g)
    JS_vals2.append(js_div(kde_points2, gaussian))
    

    if j % 10 == 0 : 
        #Real time loss plotting
        #Define figure
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25,15))

        #ax1.set_title('lr = ' + str(lr1))
        ax1.plot(np.linspace(1,j+1, len(loss_dict['train'])), loss_dict['train'],'k', label='Train')
        ax1.plot(np.linspace(1,j+1, len(loss_dict['train'])), loss_dict['val'],'r', label='Validation', alpha=0.5)
        ax1.set_ylabel('loss', fontsize = 20)
        ax1.set_xlabel('Epochs', fontsize = 20)
        ax1.set_xscale('log')
        ax1.set_ylim([-1.2,0.1])
        ax1.set_xlim([1,n_epochs])
        ax1.xaxis.set_tick_params(labelsize=20)
        ax1.yaxis.set_tick_params(labelsize=20)
        ax1.grid(True) 
        ax1.legend(fontsize = 20)
        
        #Real time latent space plotting        
        ax2.plot(g, gaussian,linewidth=3,c='k',label='Standard Gaussian')
        ax2.set_ylim([0,0.5])
        ax2.set_xlim([-5,5])
        #ax2.hist(np.array(z_[:,0]),bins=50, density=True,alpha=0.6,  label='$z_{0}$');
        #ax2.hist(np.array(z_[:,1]),bins=50, density=True,alpha=0.6, label='$z_{1}$');
        ax2.plot(g,  kde_points1, linewidth=3,label = r'$z_{0}$')
        ax2.plot(g, kde_points2, linewidth=3, label = r'$z_{1}$')
        ax2.legend(fontsize = 20)
        ax2.grid(True)
        ax2.set_ylabel('$p(z)$',fontsize = 20)
        ax2.set_xlabel('$z$',fontsize = 20)
        ax2.xaxis.set_tick_params(labelsize=20)
        ax2.yaxis.set_tick_params(labelsize=20)
        
        #Real time JS div between gaussian and latent 
        ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), JS_vals1,linewidth=3, label = r'$z_{0}$')
        ax3.plot(np.linspace(1,j+1, len(loss_dict['train'])), JS_vals2,linewidth=3, label = r'$z_{1}$')
        ax3.set_ylabel('JS Div', fontsize = 20)
        ax3.set_xlabel(r'Epochs', fontsize = 20)
        ax3.set_yscale('log')
        ax3.set_xscale('log')
        ax1.set_xlim([1,n_epochs])
        ax3.grid(True)
        ax3.xaxis.set_tick_params(labelsize=20)
        ax3.yaxis.set_tick_params(labelsize=20)
        ax3.legend(fontsize = 20)
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

def Flow_posterior(conditional, n_points = 200, device = 'cpu'):
    n = n_points**(2) 
    
    device = device
    
    flow.eval()
    flow.to(device)
    
    with torch.no_grad():
        samples = flow.sample(n, conditional=conditional.to(device))
        samples= scaler_x.inverse_transform(samples)

        n1 = n_points
        x  = np.linspace(0.1,0.9,n_points)
        y  = np.linspace(0,1,n_points)
        dx = np.diff(x)[0] ; dy = np.diff(y)[0]
        xx, yy = np.meshgrid(x, y)
        xx = xx.reshape(-1,1)
        yy = yy.reshape(-1,1)

        xy_inp = torch.from_numpy(np.concatenate([xx,yy], axis=1)).float()
        logprobx = flow.log_prob(xy_inp, conditional=conditional.to(device))
        logprobx = logprobx.numpy() 
        logprobx = logprobx.reshape(n_points,n_points)

        
        return samples, logprobx




n_points = 160
n = n_points**2
np.random.seed(1234)
Nresults =50
Nruns = 1
labels = ['H0']
priors = {}
for jj in range(1):
    if labels[jj] == 'H0':
        priors.update({f"{labels[jj]}": Uniform(20, 120, f"{labels[jj]}")})



for x in range(Nruns):
    results = []
    for ii in tqdm(range(Nresults)):
        posterior = dict()
        injections = dict()
        for key, prior in priors.items():

            if key == 'H0':
                inx = np.random.randint(10**4)
                conditional = np.array(Y_scale_val[inx,:5].reshape(1,-1))
                data = np.array(conditional[:,:4])*np.ones((n, 4))
                data_scaled = torch.from_numpy(data.astype('float32'))  
                truths=  scaler_x.inverse_transform(X_scale_val[inx,:].reshape(1,-1))[0]
                samples, _ = Flow_posterior(data_scaled, n_points = n_points)
                posterior[key] = samples[:,0] 
                injections[key] = truths[0].astype('float32').item()
  

        posterior = pd.DataFrame(dict(posterior))
        result = bilby.result.Result(
            label="test",
            injection_parameters=injections,
            posterior=posterior,
            search_parameter_keys=injections.keys(),
            priors=priors)
        results.append(result)

    fig = bilby.result.make_pp_plot(results, filename=path+folder_name+"/PP",
                              confidence_interval=(0.68, 0.90, 0.99, 0.9999))
















from poplar.distributions import VariableLimitsPowerLaw, FixedLimitsPowerLaw, UniformDistribution
from poplar.nn.networks import LinearModel, load_model
from poplar.nn.training import train, train_test_split
from poplar.nn.rescaling import ZScoreRescaler
from poplar.nn.plot import loss_plot
from poplar.selection import selection_function_from_optimal_snr
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import pandas as pd
import argparse
#pass arguments 
#Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-Name", "--Name_model", required=True,
   help="Name of data")
ap.add_argument("-device", "--device", required=True,
   help="device", default = 'cpu')
ap.add_argument("-detector", "--detector", required=True,
   help="make data from detector: OPTIONS [H1, L1, V1, combined]", nargs='+', default=[])
ap.add_argument("-run", "--run", required=True,
   help="Observing run: OPTIONS [O1, O2, O3, O4] ", default = 'O3')
ap.add_argument("-neurons", "--neurons", required=True,
   help="Set neurons per layer for MLP ", default = 128)
ap.add_argument("-layers", "--layers", required=True,
   help="Set layers for MLP ", default = 5)
ap.add_argument("-epochs", "--epochs", required=True,
   help="Set number of epochs ", default = 1000)
ap.add_argument("-batches", "--batches", required=True,
   help="Set number of batches ", default = 3)
ap.add_argument("-lr", "--learning_rate", required=True,
   help="Set learning rate ", default = 0.1e-3)
ap.add_argument("-split", "--split_data", required=True,
   help="Set training_data percentage to be used ", default = 0.8)
ap.add_argument("-loss_function", "--loss_function", required=True,
   help="either MSE or L1", default = 'MSE')





args = vars(ap.parse_args())
name = str(args['Name_model'])
device = str(args['device'])
det = list(args['detector'])
run = str(args['run'])
neurons = int(args['neurons'])
layers = int(args['layers'])
epochs = int(args['epochs'])
batches = int(args['batches'])
fraction = float(args['split_data'])
learning_rate = float(args['learning_rate'])
loss_function = str(args['loss_function'])


hyper_dictionary = {'Name': name, 'device':device, 'det':det, 'run':run, 
                   'neurons': neurons, 'layers':layers, 'epochs':epochs, 'batches':batches, 
                   'fraction': fraction, 'learning_rate':learning_rate, 'loss_function':loss_function }



print(hyper_dictionary)



if loss_function == 'MSE':
    loss_function = torch.nn.MSELoss()
elif loss_function == 'L1':
    loss_function = torch.nn.L1Loss()
else: 
    raise ValueError("Loss function {} not implemented ".format(loss_function))
    



neurons_per_layer = neurons
layers = layers
name = name + '_' + run 
for detectors in det:
    temp = '_' + detectors
    name += temp

tot_neurons = [neurons_per_layer]*layers




def ha_from_ra_gps(ra, gps):
    return (gps/(3600*24))%(86164.0905/(3600*24))*2*np.pi - ra

def read_data(batch):
    path_name =r"data_for_MLP/data_sky_theta/training/"
    data_name = "_{}_MLP_500000_det_H1_L1_V1_run_{}_approx_IMRPhenomXPHM_batch_1.csv".format(run, run, batch)

    GW_data = pd.read_csv(path_name+data_name,skipinitialspace=True)

                          
#                           usecols=['luminosity_distance', 'mass_1','mass_2',
#                   'a_1','a_2','tilt_1',
#                   'tilt_2','ra','dec',
#                    'theta_jn','phi_jl', 'phi_12',
#                    'psi','geocent_time','snr'])
    return GW_data


list_data = [] 
for i in range(1):
    list_data.append(read_data(i+1))

GW_data = pd.concat(list_data, ignore_index=True)
if det == ['H1', 'L1']:
    GW_data = GW_data[['luminosity_distance', 'mass_1','mass_2','theta_jn', 'ra', 'dec','psi', 'geocent_time',  'snr_H1', 'snr_L1', 'snr_network']]
else: 
    GW_data = GW_data[['luminosity_distance', 'mass_1','mass_2','theta_jn', 'ra', 'dec','psi', 'geocent_time',  'snr_H1', 'snr_L1','snr_V1', 'snr_network']]
    
ha = ha_from_ra_gps(np.array(GW_data.ra), np.array(GW_data.geocent_time))
GW_data['ha'] = ha

# snrs = GW_data['snr_network']
if det == 'H1':
    snrs = GW_data['snr_H1']
    print('detector = {}'.format(det))
    

    
elif det =='L1':
    snrs = GW_data['snr_L1']
    print('detector = {}'.format(det))
    
elif det =='V1': 
    snrs = GW_data['snr_V1']
    print('detector = {}'.format(det))
    
elif det =='combined':
    snrs =GW_data['snr_network']
    print('detector = {}'.format(det))
    
elif det =='all':
    snrs =GW_data[['snr_H1', 'snr_L1', 'snr_V1']]
    print('detector = {}'.format(det))
    
elif det ==['H1', 'L1']:
    snrs = GW_data[['snr_H1', 'snr_L1']]
    print('detector = {}'.format(det))    
    
elif det ==['H1', 'L1', 'V1']:
    snrs = GW_data[['snr_H1', 'snr_L1', 'snr_V1']]
    print('detector = {}'.format(det)) 
    
else: 
    raise ValueError("Detector {} does not exist".format(det))
    
    
dl = GW_data['luminosity_distance']
data = GW_data[['mass_1','mass_2', 'theta_jn', 'ha', 'dec', 'psi' ]]

xdata = torch.as_tensor(data.to_numpy(), device=device).float()
ydata = torch.as_tensor((np.array(dl).reshape(-1,1) * np.array(snrs) ), device=device).float()

rescaler = ZScoreRescaler(xdata, ydata)
xtrain, xtest, ytrain, ytest = train_test_split([xdata, ydata], fraction)
inds = fraction*len(dl)
dl_test = dl.loc[int(inds):]


# define the neural network
model = LinearModel(
    in_features=6,
    out_features=np.shape(snrs)[1],
    neurons=tot_neurons,
    activation=torch.nn.ReLU,
    rescaler=rescaler,
    name = name
)   

model.to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)


train(
    model, 
    data=[xtrain, ytrain, xtest, ytest], 
    n_epochs=epochs, 
    n_batches=batches, 
    loss_function=loss_function,
    optimiser=optimiser,
    update_every=1000,
    verbose=True,
    outdir='models/MLP_models',
    scheduler = None,#torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max = 10000, eta_min=0, last_epoch=-1, verbose=False),
    save_best= True
)


f = open('models/MLP_models/{}/hyperparameters.txt'.format(name),'w')
f.write(str(hyper_dictionary))
f.close()

train_array = model.loss_curves[0] ; test_array = model.loss_curves[1]

plt.plot(range(len(train_array)), train_array,alpha = 0.5, label = 'train')
plt.plot(range(len(train_array)), test_array,alpha =0.5,  label = 'validation')
plt.yscale('log')
plt.xscale('log')
plt.grid(True)
plt.ylabel(r'Loss', fontsize = 15)
plt.xlabel(r'Epoch', fontsize = 15)
plt.legend(loc = 'upper right', fontsize = 10)
plt.savefig('models/MLP_models/{}/loss.png'.format(name))
plt.close()



ypred = model.run_on_dataset(xtest)
snr_columns_pred = ypred.cpu().numpy() / np.array(dl_test)[:,None]
snr_columns_true = ytest.cpu().numpy() / np.array(dl_test)[:,None]


detectors = ['H1', 'L1', 'V1']

for i,det in enumerate(detectors):
# plt.hist(np.log10(abs((ypred/ytest).cpu().numpy())), bins='auto', density=True, histtype = 'step', linewidth =2)
# plt.xlabel(r'$\log_{10}(|y_{pred}/y_{test}|)$')
# plt.grid(True)
# plt.savefig('models/MLP_models/{}/ypred_vs_ytrue.png'.format(name))
# plt.close()

# snr_pred = ypred.cpu().numpy()/dl_test
# snr_true = ytest.cpu().numpy()/dl_test


# plt.hist(abs(snr_pred-snr_true), bins=100, density=1, histtype = 'step', linewidth = 2)
# plt.xlabel(r'$|\rho_{pred}-\rho_{true}|$', fontsize = 15)
# plt.yscale('log')
# plt.grid(True)
# plt.savefig('models/MLP_models/{}/snrtrue_vs_snrpred.png'.format(name))
# plt.close()


    plt.figure(figsize = (7,7))
    c = plt.scatter(x = snr_columns_pred[:,i], y = snr_columns_true[:,i], c = (abs(snr_columns_pred[:,i] - snr_columns_true[:,i])), s= 10, cmap = 'jet')
    cbar = plt.colorbar(c)
    cbar.set_label(label = r'$|\rho_{pred} - \rho_{true}|$', fontsize=20)
    plt.clim(0,4)
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    x = np.linspace(0,100_000)
    plt.plot(x,x, '--k', linewidth = 2)
    plt.xlabel(r'$\rho_{pred}$', fontsize = 15)
    plt.ylabel(r'$\rho_{true}$', fontsize = 15)
    plt.grid(True)
    plt.savefig('models/MLP_models/{}/true_vs_spred_line_{}.png'.format(name, det))
    plt.close()


# plt.figure(figsize = (7,7))
# c = plt.scatter(x = snr_pred, y = snr_true, c = (abs(snr_pred - snr_true)), s= 10, cmap = 'jet')
# cbar = plt.colorbar(c)
# cbar.set_label(label = r'$|\rho_{pred} - \rho_{true}|$', fontsize=20)
# plt.clim(0,4)
# plt.xlim([8, 15])
# plt.ylim([8, 15])
# x = np.linspace(0,100_000)
# plt.plot(x,x, '--k', linewidth = 2)
# plt.xlabel(r'$\rho_{pred}$', fontsize = 15)
# plt.ylabel(r'$\rho_{true}$', fontsize = 15)
# plt.grid(True)
# plt.savefig('models/MLP_models/{}/true_vs_spred_line_zoom.png'.format(name))
# plt.close()

    point = 6
    lim_snr_inx = np.where((snr_columns_true[:,i]>point - 0.1) & (snr_columns_true[:,i]<point + 0.1))[0]
    plt.hist(np.array(snr_columns_pred[:,i])[lim_snr_inx], bins = 'auto', histtype = 'step', linewidth = 3, label = 'SNR_pred')
    plt.axvline(x = point, color = 'r', label = 'SNRth = {} +/- 0.1'.format(point))
    plt.xlabel(r'$\rho_{pred}$', fontsize = 15 )
    plt.grid(True)
    plt.legend(loc = 'upper right')
    plt.savefig('models/MLP_models/{}/snrTH_distribution_{}.png'.format(name, det))
    plt.close()
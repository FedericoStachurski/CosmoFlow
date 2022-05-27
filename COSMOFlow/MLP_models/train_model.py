from torch import nn
import torch
import numpy as np
from MultiLayerPerceptron.nn.model_train_test import model_train_test
from MultiLayerPerceptron.nn.model_creation import create_mlp
import pandas as pd



def read_data(batch):
    path_name ="data_for_MLP/training/"
    data_name = "_data_{}_batch_{}.csv".format(25000, batch)
    GW_data = pd.read_csv(path_name+data_name,skipinitialspace=True, usecols=[ 'dl', 'm1z', 'm2z', 'a1', 'a2', 'tilt1', 'tilt2', 'RA', 'dec', 'theta_jn', 'phi_jl', 'phi_12', 'psi', 'snr'])
    return GW_data

list_data = [] 
for i in range(6):
    list_data.append(read_data(i+1))


GW_data = pd.concat(list_data)
GW_data = GW_data[[ 'dl', 'm1z', 'm2z', 'a1', 'a2', 'tilt1', 'tilt2', 'RA', 'dec', 'theta_jn', 'phi_jl', 'phi_12', 'psi', 'snr']]
if __name__ == '__main__':
    device = "cuda:2"

    #['H0','z', dl, m1, m2, chi1, chi2, inc, RA, dec 'Mz', 'inc', 'q', 'SNR']
    train_inds = [0,1,2,3,4,5,6,7,8,9,10, 11, 12]
    test_inds = [13]
    print('Showing DATA')
    print(GW_data.iloc[:,:13])
  
    traindata = GW_data #pd.read_csv(fp.format('samp_dataframe.cdata_for_MLPsv'))
    xtrain = traindata.iloc[:,train_inds].to_numpy()
    ytrain = traindata.iloc[:,test_inds].to_numpy()
    
    
#     xtrain = xtrain[np.where(~np.isnan(ytrain[:,1]))[0],:]
#     ytrain = ytrain[np.where(~np.isnan(ytrain)[:,1])[0],:]

    
    inds = np.arange(ytrain.shape[0])
    np.random.shuffle(inds)
    xtrain = xtrain[inds,:]
    ytrain = ytrain[inds,:]
        
    cut = 0.85
    xtest = xtrain[int(cut*xtrain.shape[0]):,:]
    ytest = ytrain[int(cut*ytrain.shape[0]):,:]
    xtrain = xtrain[:int(cut*xtrain.shape[0]),:]
    ytrain = ytrain[:int(cut*ytrain.shape[0]),:]

    print('LENGTH Training:' +str(ytrain.shape[0]))
    print('LENGTH Validation:' +str(ytest.shape[0]))
    print('TRAINING: '+str(cut)+' ; VALIDATION: '+str(1-cut))
    
    in_features = 13
    out_features = 1
    layers = 3
    neurons = np.array(np.ones(layers,dtype=np.int32)*256).tolist()#[256,256,256,256,256]
    activation = nn.SiLU
    out_activation=None
    model = create_mlp(input_features=in_features,output_features=out_features,neurons=neurons,layers=layers,activation=activation,
                       out_activation=out_activation, device=device, model_name='SNR_approxiamator',use_bn=False)

    data = [xtrain, ytrain, xtest, ytest]

    loss_function = nn.L1Loss()
    LR = 0.5*1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    sched = None#torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=5e-5,max_lr=1e-3, step_size_up=5000,mode='triangular2', cycle_momentum=False)
    
    model_train_test(data, model, device,n_epochs=50000, n_batches=1, loss_function=loss_function,optimizer=optimizer, verbose=True,update_every=100, n_test_batches=1,save_best=True, scheduler=sched)

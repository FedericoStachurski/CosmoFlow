from MLP_models.MultiLayerPerceptron.validate import run_on_dataset
from MLP_models.MultiLayerPerceptron.nn.model_creation import load_mlp


import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
device = 'cpu'

model_name = 'SNR_approxiamator'
mlp = load_mlp(model_name, device, get_state_dict=True).to(device)
mlp.eval()



def SNR_from_MLP(GW_data):


    df = GW_data

    x_inds = [0,1,2]

    xdata = df.iloc[:,x_inds].to_numpy()


    xmeanstd = np.load(f'models/{model_name}/xdata_inputs.npy')

    net_out, time_tot, time_p_point = run_on_dataset(mlp,xdata,label_dim = None, 
                                                        device=device,y_transform_fn=None,runtime=True)

    pred = net_out


    pred[pred<0]=0

    inx = np.where(pred != 0)


    pred = pred[inx]

    snr_out = pred
    return snr_out


import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
import argparse


#pass arguments 
# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-Name", "--Name_folder", required=True,
   help="Name of the folder of FLOW model")
ap.add_argument("-npoints_filter ", "--npoints_filter", required=False, default = 20,
   help="Name of the folder of FLOW model")
ap.add_argument("-log_epochs ", "--log_epochs", required=False, default = 1,
   help="Name of the folder of FLOW model")


args = vars(ap.parse_args())
Name = str(args['Name_folder'])
# data = str(args['data_folder'])
npoints_filter = int(args['npoints_filter'])
log = int(args['log_epochs'])


flow = 'trained_flows_and_curves/'+Name
#load_data
kl_data = pd.read_pickle(flow+'/kl_data.pickle')
loss_dict = pd.read_pickle(flow+'/loss_data.pickle')

# print(kl_data)
# print(loss)

# print(kl_data['KL_vals1'])

n_epochs = len(loss_dict['train'])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))

#ax1.set_title('lr = ' + str(lr1))

ax1.plot(loss_dict['train'],'k', label='Train', linewidth = 3)
ax1.plot(loss_dict['val'],'r', label='Validation', alpha=0.4, linewidth = 3)
ax1.set_ylabel('loss', fontsize = 20)
ax1.set_xlabel('Epochs', fontsize = 20)


if log == 1: 
    ax1.set_xscale('log')
    
ax1.set_ylim([np.min(loss_dict['train'])-0.1,np.max(loss_dict['train'])*1.25])
ax1.set_xlim([1,n_epochs])
ax1.xaxis.set_tick_params(labelsize=20)
ax1.yaxis.set_tick_params(labelsize=20)
ax1.grid(True) 
ax1.legend(fontsize = 20)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

kls = ['KL_vals1', 'KL_vals2', 'KL_vals3', 'KL_vals4', 'KL_vals5']
for i, kl_key in enumerate(kls):
    l, = ax2.plot(kl_data[str(kl_key)],linewidth=1,alpha = 0.1, )
    color = l.get_color()
    # ax2.plot(savgol_filter(np.array(kl_data[str(kl_key)]),  51, 3), linewidth=2,alpha = 0.7 , color = color, label = r'$z{}$'.format(i))
    ax2.plot(smooth(np.array(kl_data[str(kl_key)]), npoints_filter), linewidth=2,alpha = 0.7 , color = color, label = r'$z{}$'.format(i))

ax2.set_xlim([1,n_epochs])
ax2.set_ylim([1e-4,10])
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.xaxis.set_tick_params(labelsize=20)
ax2.yaxis.set_tick_params(labelsize=20)
ax2.set_ylabel('KLdiv', fontsize = 20)
ax2.set_xlabel('Epochs', fontsize = 20)
ax2.grid(True) 
ax2.legend(fontsize = 20)


fig.tight_layout()
fig.savefig(flow+'/training.png', dpi = 50 )

plt.close('all')   # Clear figure

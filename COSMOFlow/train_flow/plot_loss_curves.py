import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
import argparse
import corner

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
latent_samples = pd.read_pickle(flow+'/latent_data.pickle')
print('Flow is at EPOCH = {}'.format(len(loss_dict['train'])))
print()
# print(loss_dict)

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
    
ax1.set_ylim([np.min(loss_dict['train'])-0.1,np.max(loss_dict['train'])+0.1])
# ax1.set_ylim([3.75,5])
ax1.set_xlim([1,n_epochs])
ax1.xaxis.set_tick_params(labelsize=20)
ax1.yaxis.set_tick_params(labelsize=20)
ax1.grid(True) 
ax1.legend(fontsize = 30)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

kls = ['KL_vals1', 'KL_vals2', 'KL_vals3', 'KL_vals4', 'KL_vals5',
       'KL_vals6', 'KL_vals7', 'KL_vals8', 'KL_vals9', 'KL_vals10', 'KL_vals11', 'KL_vals12', 'KL_vals13', 'KL_vals14']
for i, kl_key in enumerate(kls):
    l, = ax2.plot(kl_data[str(kl_key)],linewidth=1,alpha = 0.25, )
    color = l.get_color()
    # ax2.plot(savgol_filter(np.array(kl_data[str(kl_key)]),  51, 3), linewidth=2,alpha = 0.7 , color = color, label = r'$z{}$'.format(i))
    ax2.plot(smooth(np.array(kl_data[str(kl_key)]), npoints_filter), linewidth=2,alpha = 0.7 , color = color, label = r'$z{}$'.format(i))

ax2.set_xlim([1,n_epochs])
ax2.set_ylim([1e-4,1])
# ax2.set_xscale('log')
if log == 1: 
    ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.xaxis.set_tick_params(labelsize=20)
ax2.yaxis.set_tick_params(labelsize=20)
ax2.set_ylabel('KLdiv', fontsize = 20)
ax2.set_xlabel('Epochs', fontsize = 20)
ax2.grid(True) 
ax2.legend(fontsize = 13)


fig.tight_layout()
fig.savefig(flow+'/training.png', dpi = 50 )

plt.close('all')   # Clear figure


##### latent_space
df_latent = pd.DataFrame({
    'z1': latent_samples['z_samples'][:, 0],
    'z2': latent_samples['z_samples'][:, 1],
    'z3': latent_samples['z_samples'][:, 2],
    'z4': latent_samples['z_samples'][:, 3],
    'z5': latent_samples['z_samples'][:, 4],

    'z6': latent_samples['z_samples'][:, 5],
    'z7': latent_samples['z_samples'][:, 6],
    'z8': latent_samples['z_samples'][:, 7],
    'z9': latent_samples['z_samples'][:, 8],
    'z10': latent_samples['z_samples'][:, 9],

    'z11': latent_samples['z_samples'][:, 10],
    'z12': latent_samples['z_samples'][:, 11],
    'z13': latent_samples['z_samples'][:, 12],
    'z14': latent_samples['z_samples'][:, 13],
})

gaussian = np.random.normal(np.zeros(14), np.ones(14), size = (10000,14))

df_gaussian = pd.DataFrame({
    'z1': gaussian[:, 0],
    'z2': gaussian[:, 1],
    'z3': gaussian[:, 2],
    'z4': gaussian[:, 3],
    'z5': gaussian[:, 4],
    'z6': gaussian[:, 5],
    'z7': gaussian[:, 6],
    'z8': gaussian[:, 7],
    'z9': gaussian[:, 8],
    'z10': gaussian[:, 9],
    'z11': gaussian[:, 10],
    'z12': gaussian[:, 11],
    'z13': gaussian[:, 12],
    'z14': gaussian[:, 13],
})

from scipy.stats import gaussian_kde
from scipy.integrate import quad

# Generate sample data
np.random.seed(0)  # For reproducibility
i = 3
kls = []
print()
print('Showing KL divergences in 1D latent space')
for i in range(14):
    # if i+1 == 6:
    #     break
    # Estimate the PDFs using KDE
    pdf_p = gaussian_kde(np.array(df_latent[['z'+str(i+1)]]).T)
    pdf_q = gaussian_kde(np.array(df_gaussian[['z'+str(i+1)]]).T)
    
    # Define the KL divergence calculation
    def kl_divergence(p, q, start=-5, end=5):
        # Numerical integration of p(x) * log(p(x) / q(x))
        return quad(lambda x: p(x) * np.log(p(x) / q(x)), start, end)[0]
    
    # Calculate the KL divergence D_KL(P || Q)
    # Note: We limit the integration bounds to a reasonable interval for numerical stability
    kl_div_pq = kl_divergence(pdf_p.evaluate, pdf_q.evaluate, start = -5, end = 5)
    print('z{} KL: {}'.format(i+1,kl_div_pq))
    kls.append(kl_div_pq)

print(np.shape(df_latent))


# Plot the corner plot with Gaussian overlays
fig = corner.corner(df_latent, range=[(-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5),
                                     (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5),
                                     (-5, 5), (-5, 5), (-5, 5), (-5, 5)],
                    labels=[r'$z_{0}$', r'$z_{1}$', r'$z_{2}$', r'$z_{3}$', r'$z_{4}$',
                           r'$z_{5}$', r'$z_{6}$', r'$z_{7}$', r'$z_{8}$', r'$z_{9}$',
                           r'$z_{10}$', r'$z_{11}$', r'$z_{12}$', r'$z_{13}$', r'$z_{14}$'], label_kwargs={'fontsize': 25},
                    hist_bin_factor=2, smooth = True,
                    hist_kwargs={"density": True}) # ensure normalization)
fig2 = corner.corner(gaussian, range=[(-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5),
                                     (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5),
                                     (-5, 5), (-5, 5), (-5, 5), (-5, 5)],
                    labels=[r'$z_{0}$', r'$z_{1}$', r'$z_{2}$', r'$z_{3}$', r'$z_{4}$',
                           r'$z_{5}$', r'$z_{6}$', r'$z_{7}$', r'$z_{8}$', r'$z_{9}$',
                           r'$z_{10}$', r'$z_{11}$', r'$z_{12}$', r'$z_{13}$', r'$z_{14}$'], label_kwargs={'fontsize': 25},
                    hist_bin_factor=2, fig = fig, smooth = True, plot_datapoints=False,plot_density=False,
                    hist_kwargs={"density": True}, color = 'red') # ensure normalization)

blue_line = plt.Line2D([0], [0], linewidth = 10,  color='black', label='Latent Space')
red_line = plt.Line2D([0], [0],  linewidth = 10, color='red', label='Multivariate Normal')
JSdl = plt.Line2D([0],[0], linestyle = 'none', label = r'$z_{0} \:\:$' + f"KL= {kls[0]*1000:.5f} millinats")
JSra = plt.Line2D([0],[0], linestyle = 'none', label = r'$z_{1} \:\:$' + f"KL= {kls[1]*1000:.5f} millinats")
JSdec = plt.Line2D([0],[0], linestyle = 'none', label = r'$z_{2}\:\:$' + f"KL= {kls[2]*1000:.5f} millinats")
JSm1 = plt.Line2D([0],[0], linestyle = 'none', label = r'$z_{3} \:\:$' + f"KL= {kls[3]*1000:.5f} millinats")
JSm2 = plt.Line2D([0],[0], linestyle = 'none', label = r'$z_{4} \:\:$' + f"KL= {kls[4]*1000:.5f} millinats")

JS5 = plt.Line2D([0],[0], linestyle = 'none', label = r'$z_{5} \:\:$' + f"KL= {kls[5]*1000:.5f} millinats")
JSr6 = plt.Line2D([0],[0], linestyle = 'none', label = r'$z_{6} \:\:$' + f"KL= {kls[6]*1000:.5f} millinats")
JS7 = plt.Line2D([0],[0], linestyle = 'none', label = r'$z_{7}\:\:$' + f"KL= {kls[7]*1000:.5f} millinats")
JS8 = plt.Line2D([0],[0], linestyle = 'none', label = r'$z_{8} \:\:$' + f"KL= {kls[8]*1000:.5f} millinats")
JS9 = plt.Line2D([0],[0], linestyle = 'none', label = r'$z_{9} \:\:$' + f"KL= {kls[9]*1000:.5f} millinats")

JS10 = plt.Line2D([0],[0], linestyle = 'none', label = r'$z_{10} \:\:$' + f"KL= {kls[10]*1000:.5f} millinats")
JS11 = plt.Line2D([0],[0], linestyle = 'none', label = r'$z_{11} \:\:$' + f"KL= {kls[11]*1000:.5f} millinats")
JS12 = plt.Line2D([0],[0], linestyle = 'none', label = r'$z_{12}\:\:$' + f"KL= {kls[12]*1000:.5f} millinats")
JS13 = plt.Line2D([0],[0], linestyle = 'none', label = r'$z_{13} \:\:$' + f"KL= {kls[13]*1000:.5f} millinats")


for ax in fig2.get_axes():
    ax.tick_params(axis='both', which='major', labelsize=15)  # Adjust as needed
    ax.tick_params(axis='both', which='minor', labelsize=10)  # Adjust as needed

fig2.legend(handles=[blue_line, red_line, JSdl, JSra, JSdec, JSm1, JSm2,
                     JS5, JSr6, JS7 ,JS8, JS9, JS10, JS11, JS12, JS13 ], loc=[0.55,0.65], frameon=True, fontsize =30)
plt.savefig(flow+'/latent_space.png', dpi = 300)
plt.show()





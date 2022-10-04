from MultiLayerPerceptron.validate import run_on_dataset
from MultiLayerPerceptron.nn.model_creation import load_mlp
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
device = 'cpu'
import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')

model_name = 'SNR_approxiamator_full_para_HA_v8'
mlp = load_mlp(model_name, device, get_state_dict=True).to(device)
mlp.eval()



path_name =r"data_for_MLP/data_sky_theta/testing/"
data_name = "_data_full_para_uniform_snr.csv".format(1)
GW_data = pd.read_csv(path_name+data_name,skipinitialspace=True, usecols=['dl','m1z','m2z',
                                                                              'a1','a2','tilt1',
                                                                              'tilt2','RA',
                                                                              'dec','thteta_jn','phi_jl', 'phi_12', 
                                                                              'polarization','geo_time','snr'])


def HA(time, RA):
    def time2angle(time):
        time = time / 3600.0
        return time*(15/1)*(np.pi/180)
    LHA = (time2angle(time) - RA)*(180/np.pi)



    return LHA


ha = HA(GW_data.geo_time, GW_data.RA - np.pi)
#GW_data['HA'] = ha

GW_data = GW_data[['dl','m1z','m2z',
                  'a1','a2','tilt1',
                  'tilt2','RA','geo_time',
                  'dec','thteta_jn','phi_jl', 'phi_12',
                   'polarization','snr']]


df = GW_data


x_inds = [0, 1,2, 3, 4, 5, 6, 7, 8, 9,10, 11, 12, 13]
y_inds = [14]

xdata = df.iloc[:,x_inds].to_numpy()
ydata = df.iloc[:,y_inds].to_numpy()

print('Total:',ydata.size)

xmeanstd = np.load(f'models/{model_name}/xdata_inputs.npy')
ymeanstd = np.load(f'models/{model_name}/ydata_inputs.npy')

net_out, time_tot, time_p_point = run_on_dataset(mlp,xdata,label_dim = None, device=device,y_transform_fn=None,runtime=True)

print('Time taken for network (in total, per point): ',f'({time_tot:.3e},{time_p_point:.3e})')

truth = ydata
pred = net_out
pred = np.exp(pred)
#pred[pred<0]=0

truth = np.array(truth.T)[0]

Path(f'models/{model_name}/accuracy_plots').mkdir(parents=True, exist_ok=True)



#plot accuracies
x = np.linspace(0,1000, 100)

fig1 = plt.figure()
cm = plt.cm.get_cmap('plasma')
sc = plt.scatter(pred, truth, c=np.log10(df.dl), vmin=2, vmax=4, s=5, cmap=cm)
cbar = plt.colorbar(sc)
cbar.set_label('$log_{10}(D_{L}[Mpc])$ ', rotation=270,  labelpad=15)
plt.plot(x,x, 'k', linewidth=2, alpha = 0.9)

plt.xlim([2,100])
plt.ylim([2,100])
plt.ylabel('TRUE')
plt.xlabel('PRED')
fig1.savefig(f'models/{model_name}/accuracy_plots/TRUEvsPRED.png', bbox_inches = 'tight', dpi = 300)


fig1 = plt.figure()
sc = plt.scatter(pred, truth, c=np.log10(df.dl), vmin=2, vmax=4, s=25, cmap=cm)
cbar = plt.colorbar(sc)
cbar.set_label('$log_{10}(D_{L}[Mpc])$', rotation=270, labelpad=15)
plt.plot(x,x, 'k', linewidth=2, alpha = 0.9)
plt.xlim([8,12])
plt.ylim([8,12])
plt.ylabel('TRUE')
plt.xlabel('PRED')
fig1.savefig(f'models/{model_name}/accuracy_plots/TRUEvsPRED_zoom.png', bbox_inches = 'tight', dpi = 300)


fig2 = plt.figure()
diff = np.array(np.array(pred - truth).T)


# plt.scatter(truth, np.abs(diff), s= 3, color = 'black')#
plt.hist(diff, bins = 'auto', edgecolor = 'blue', density  = 0)
# plt.ylim([0,20])
plt.xlim([-30,30])
plt.xlabel('pred - truth')
plt.ylabel('Count')
fig2.savefig(f'models/{model_name}/accuracy_plots/TRUEvsPRED_difference.png', bbox_inches = 'tight', dpi =300)



fig3 = plt.figure()
ratio = np.array(pred / truth)
plt.hist(ratio, bins = 150, edgecolor = 'blue', density  = 0 )
plt.axvline(x = 1, color = 'red', linewidth = 0.5)
plt.xlabel('Pred / Truth')
plt.ylabel('Count')
fig3.savefig(f'models/{model_name}/accuracy_plots/TRUEvsPRED_ratio.png', bbox_inches = 'tight', dpi = 300)


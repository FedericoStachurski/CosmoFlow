from MultiLayerPerceptron.validate import run_on_dataset
from MultiLayerPerceptron.nn.model_creation import load_mlp
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
device = 'cpu'

model_name = 'SNR_approxiamator_2'
mlp = load_mlp(model_name, device, get_state_dict=True).to(device)
mlp.eval()


path_name ="data_for_MLP/data_dl_mm/testing_data/"
data_name = "_data_{}.csv".format(2500)
GW_data = pd.read_csv(path_name+data_name,skipinitialspace=True, usecols=[ 'dl', 'm1z', 'm2z', 'snr'])
df = GW_data

x_inds = [0,1,2]
y_inds = [3]

xdata = df.iloc[:,x_inds].to_numpy()
ydata = df.iloc[:,y_inds].to_numpy()

print('Total:',ydata.size)

xmeanstd = np.load(f'models/{model_name}/xdata_inputs.npy')
ymeanstd = np.load(f'models/{model_name}/ydata_inputs.npy')

net_out, time_tot, time_p_point = run_on_dataset(mlp,xdata,label_dim = None, device=device,y_transform_fn=None,runtime=True)

print('Time taken for network (in total, per point): ',f'({time_tot:.3e},{time_p_point:.3e})')

truth = ydata
pred = net_out
pred[pred<0]=0

truth = np.array(truth.T)[0]

Path(f'models/{model_name}/accuracy_plots').mkdir(parents=True, exist_ok=True)



#plot accuracies
x = np.linspace(0,1000, 100)

fig1 = plt.figure()
plt.loglog(pred, truth, '.k', markersize=5)
plt.plot(x,x, 'r', linewidth=2, alpha = 0.5)
plt.xlim([0.01,1000])
plt.ylim([0.01,1000])
plt.ylabel('TRUE')
plt.xlabel('PRED')
fig1.savefig(f'models/{model_name}/accuracy_plots/TRUEvsPRED.png', bbox_inches = 'tight', dpi = 300)


fig2 = plt.figure()
diff = np.array(np.array(pred - truth).T)


plt.scatter(truth, np.abs(diff), s= 3, color = 'black')#
# plt.hist(np.array(diff), bins = 150, edgecolor = 'blue', density  = 0)
# plt.ylim([0,20])
# plt.xlim([0,100])
plt.ylabel('|pred - truth|')
plt.xlabel('True SNR')
fig2.savefig(f'models/{model_name}/accuracy_plots/TRUEvsPRED_difference.png', bbox_inches = 'tight', dpi =300)



fig3 = plt.figure()
ratio = np.array(pred / truth)
plt.hist(ratio, bins = 150, edgecolor = 'blue', density  = 0 )
plt.axvline(x = 1, color = 'red', linewidth = 0.5)
plt.xlabel('Pred / Truth')
plt.ylabel('Count')
fig3.savefig(f'models/{model_name}/accuracy_plots/TRUEvsPRED_ratio.png', bbox_inches = 'tight', dpi = 300)


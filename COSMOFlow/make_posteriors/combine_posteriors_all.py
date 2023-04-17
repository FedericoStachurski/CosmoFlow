import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 

import argparse
# Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-Folder", "--Name_folder", required=True,
   help="Name of the folder to get the GW_posteriors")
args = vars(ap.parse_args())
Folder = str(args['Name_folder'])

folder = Folder
data = os.listdir(folder+'/posteriors/')
# data_no_w = os.listdir(folder+'/posteriors_no_w/')
O3_H0_posteriors = os.listdir(folder+'/O3_H0_post/')

files = []



event_data_saved = [] 
event_data_saved_no_w = []
O3_post = []
print('There are {} events in the folder'.format(len(data)))
for file in data:
    if file.endswith('.txt'): #and (file != 'GW190521_030229.txt') and (file != 'GW190728_064510.txt') and (file != 'GW190602_175927.txt') and (file != 'GW190412_053044.txt'):

        files.append(file)
        event_data_saved.append(np.loadtxt(folder+'/posteriors/'+file))

# for file in data_no_w:
#     if file.endswith('.txt'):

#         event_data_saved_no_w.append(np.loadtxt(folder+'/posteriors_no_w/'+file))        
        
        
for file in O3_H0_posteriors:
    if file.endswith('.txt'):

        O3_post.append(np.loadtxt(folder+'/O3_H0_post/'+file))        
print(files)    




# short_names = ['GW150914', 'GW151226', 'GW170104', 'GW170608', 'GW170809', 'GW170814', 'GW170818', 'GW170823', 'GW190412', 'GW190425', 'GW190521','GW190814' ]    
short_names = ['GW170809', 'GW170814', 'GW170818']    
####EMPTY
i = 0
path_empty = '../../../o3-cosmology/gwcosmo_results/mature_circulation_material/results/Mu_g_32.27_Mmax_112.5_band_K_Lambda_4.59_empty/'
posts_empty = [] 
for name in short_names:

    path_file_npz = path_empty + name+'/'+name+'.npz' 
    data_empty = np.load(path_file_npz, allow_pickle=True)
    data_empty = data_empty['arr_0']
    H0_grid_empty = data_empty[0]
    posterior_empty = data_empty[2]
    posts_empty.append(posterior_empty)
    i+=1
#     if i == 2:
#         break




Npoints = len(event_data_saved[0])
#H0vec = np.linspace(30,120,Npoints)
H0vec = np.linspace(20,140,Npoints)
labelsfig = plt.figure(figsize=(15,10))
values= np.ones(Npoints)
values_no_w= np.ones(Npoints)
valuesO3= np.ones(Npoints)
valuesO3_empty = np.ones(len(posterior_empty)) 
dH = np.diff(H0vec)[0]
r = len(event_data_saved)

for i in range(r):
    
    like = event_data_saved[i]
#     like_no_w  = event_data_saved_no_w[i]
    plt.plot(H0vec,like/np.sum(like*dH), alpha=0.4,linewidth = 3,  label = files[i][:13])
    values *= like
#     values_no_w *= like_no_w
    
    like_O3 = O3_post[i]
    #plt.plot(H0vec,like/np.sum(like*dH), alpha=0.4,linewidth = 3,  label = files[i][:13])
    valuesO3 *= like_O3    
    #post = values / np.sum( values*dH)

    valuesO3_empty *= posts_empty[i]

        

        
planck_h = 0.6774*100
sigma_planck_h = 0.0062*100
riess_h = 0.7324*100
sigma_riess_h = 0.0174*100



post = values / np.sum( values*dH)
# post_no_w = values_no_w / np.sum( values_no_w*dH)


H0_O3 = np.linspace(20,140, len(valuesO3))
postO3 = valuesO3 / np.sum( valuesO3*np.diff(H0_O3)[0])

postO3_empty = valuesO3_empty / np.sum( valuesO3_empty*np.diff(H0_grid_empty)[0])

ymin = 0.0005
ymax = 1.5*max(post)
ymaxO3 = 1.5*max(postO3)

if ymaxO3 > ymax:
    ymax = ymaxO3


c=sns.color_palette('colorblind')

#plt.plot(H0vec, posterior_O3,'--b', linewidth = 5, label = 'GWcosmo O3 posterior')
plt.axvline(planck_h,label='Planck',color=c[4])
plt.fill_betweenx([ymin,ymax],planck_h-2*sigma_planck_h,planck_h+2*sigma_planck_h,color=c[4],alpha=0.2)
plt.axvline(riess_h,label='SH0ES',color=c[2])
plt.fill_betweenx([ymin,ymax],riess_h-2*sigma_riess_h,riess_h+2*sigma_riess_h,color=c[2],alpha=0.2)
plt.axvline(70,ls='--', color='k',alpha=0.8, label = r'$H_0 = 70$ (km s$^{-1}$ Mpc$^{-1}$)')

plt.plot(H0vec,post, '--k', alpha=1, linewidth=6, label = '$p(H_{0} | \mathbf{h})$, posterior')
# plt.plot(H0vec,post_no_w, '--r', alpha=1, linewidth=6, label = '$p(H_{0} | \mathbf{h})$, posterior unweighted')
plt.plot(H0vec,postO3, '--b', alpha=1, linewidth=6, label = '$p(H_{0} | \mathbf{h})$, O3 gwcosmo')
plt.plot(H0_grid_empty,postO3_empty, '--g', alpha=1, linewidth=6, label = '$p(H_{0} | \mathbf{h})$, O3 gwcosmo ***EMPTY')
plt.xlim([20,140])
plt.ylim([ymin,ymax])
plt.legend(loc = 'best', fontsize = 11, ncol = 5)
plt.grid(True, alpha = 0.5)

plt.xlabel(r'$H_{0} \: [km \: s^{-1} \: Mpc^{-1}]$',fontsize = 20)
plt.ylabel(r'$p(H_{0}) \: [km^{-1} \: s \: Mpc] $',fontsize = 20)
plt.savefig(Folder+'/combine_posterior.png') 

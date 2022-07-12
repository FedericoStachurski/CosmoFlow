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
data = os.listdir(folder)
files = []

event_data_saved = [] 
for file in data:
    if file.endswith('.txt'):
#         if not file.endswith('629.txt'):
        files.append(file)
        event_data_saved.append(np.loadtxt(folder+'/'+file))
print(files)    
    
Npoints = len(event_data_saved[0])
#H0vec = np.linspace(30,120,Npoints)
H0vec = np.linspace(30,110,Npoints)
labelsfig = plt.figure(figsize=(15,10))
values= np.ones(Npoints)
dH = np.diff(H0vec)[0]
r = len(event_data_saved)
for i in range(r):
    
    like = event_data_saved[i]
    plt.plot(H0vec,like/np.sum(like*dH), alpha=0.8,linewidth = 3,  label = files[i][:13] )
    values *= like
    #post = values / np.sum( values*dH)
    

planck_h = 0.6774*100
sigma_planck_h = 0.0062*100
riess_h = 0.7324*100
sigma_riess_h = 0.0174*100



post = values / np.sum( values*dH)

ymin = 0.0005
ymax = 1.5*max(post)


c=sns.color_palette('colorblind')
plt.axvline(planck_h,label='Planck',color=c[4])
plt.fill_betweenx([ymin,ymax],planck_h-2*sigma_planck_h,planck_h+2*sigma_planck_h,color=c[4],alpha=0.2)
plt.axvline(riess_h,label='SH0ES',color=c[2])
plt.fill_betweenx([ymin,ymax],riess_h-2*sigma_riess_h,riess_h+2*sigma_riess_h,color=c[2],alpha=0.2)
plt.axvline(70,ls='--', color='k',alpha=0.8, label = r'$H_0 = 70$ (km s$^{-1}$ Mpc$^{-1}$)')

plt.plot(H0vec,post, '--k', alpha=1, linewidth=6, label = '$p(H_{0} | \mathbf{h})$, posterior')

plt.xlim([30,110])
plt.ylim([ymin,ymax])
plt.legend(loc = 'best', fontsize = 15, ncol = 4)
plt.grid(True, alpha = 0.5)

plt.xlabel(r'$H_{0} \: [km \: s^{-1} \: Mpc^{-1}]$',fontsize = 20)
plt.ylabel(r'$p(H_{0}) \: [km^{-1} \: s \: Mpc] $',fontsize = 20)
plt.savefig(Folder+'/combine_posterior.png') 

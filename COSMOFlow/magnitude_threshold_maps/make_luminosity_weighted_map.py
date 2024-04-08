import numpy as np 
import multiprocessing
import healpy as hp
from tqdm import tqdm 
import pandas as pd
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from cosmology_functions.schechter_functions import Schechter_function
from cosmology_functions import priors, cosmology, utilities
from cosmology_functions.z_parameters_dist import RedshiftGW_fast_z_para
from cosmology_functions import cosmology as csm
import matplotlib.pyplot as plt


H0 = 100
threads = 10
NSIDE = 32
band = 'K'
Npix = hp.nside2npix(NSIDE)
zmax = 1.6
zmin = 0.0000001
def load_cat_by_pix(pix):
    loaded_pix = pd.read_csv('/data/wiay/federico/PhD/cosmoflow/COSMOFlow/pixelated_catalogs/GLADE+_pix_NSIDE_{}/pixel_{}'.format(NSIDE,pix))
    #loaded_pix = loaded_pix.dropna()
    return loaded_pix

with multiprocessing.Pool(threads) as p:
    catalog_pixelated = list(tqdm(p.imap(load_cat_by_pix,np.arange(Npix)), total = Npix, desc = 'Loading mth map, NSIDE = {}'.format(NSIDE)))
map_mth = np.loadtxt('/data/wiay/federico/PhD/cosmoflow/COSMOFlow/magnitude_threshold_maps/NSIDE_{}_mth_map_GLADE_{}.txt'.format(NSIDE,band))
sch_fun = Schechter_function(band) #initiate luminosity functions class 

#### compute luminosities
L_out_map = []
L_in_map = []
L_in_map_theory = []
w_out_map = []
w_in_map = [] 
Ltot = []
# 
for i in tqdm(range(hp.nside2npix(NSIDE))):
    mth = map_mth[i]  
    Lout = sch_fun.Lout_theory(NSIDE,mth, H0, zmax, zmin)    
    Ltot_theory = sch_fun.Ltot_theory(NSIDE,mth, H0, zmax, zmin)  
    if mth > 0.0 :
        Lin_theory = sch_fun.Lin_theory(NSIDE,mth, H0, zmax, zmin)
        L_in_map_theory.append(Lin_theory)
        L_out_map.append(Lout)

        pixel = catalog_pixelated[i][['RA','dec','mK','z']]
        pixel = pixel[pixel.z > 0 ]
        pixel[pixel.mK < mth ]
        pixel = pixel.dropna()
        if len(pixel) == 0:
            Lpix = 0.0

        else:     
            Lpix = sch_fun.M2L(csm.abs_M(pixel.mK, csm.fast_z_to_dl_v2(pixel.z, 100))+ 5*np.log10(H0/100))
            Lpix = np.sum(Lpix)
            L_in_map.append(Lpix)

        w_out_map.append(Lout/(Lout+Lpix))
        w_in_map.append(Lpix/(Lout+Lpix))
        Ltot.append(Lout+Lpix)
        
    else: 
        w_out_map.append(1.0)
        Ltot.append(Ltot_theory)
        L_in_map_theory.append(0.0)
        L_in_map.append(0.0)
        w_in_map.append(0.0)

### Save total luminosity 
pixel_weight = np.array(Ltot)/np.max(np.array(Ltot))
np.savetxt('/data/wiay/federico/PhD/cosmoflow/COSMOFlow/magnitude_threshold_maps/luminosity_weighted_map/NSIDE_'+str(NSIDE)+'_Luminosity_weight_PIXEL_map_GLADE_'+band+'_zmax'+str(zmax)+'.txt', pixel_weight)
hp.mollview(pixel_weight, rot = (180,0,0), title = 'Ltot')
hp.graticule(True)
plt.savefig('/data/wiay/federico/PhD/cosmoflow/COSMOFlow/magnitude_threshold_maps/luminosity_weighted_map/NSIDE_Luminosity_weight_map_GLADE_'+band+'_zmax'+str(zmax)+'.png')

### Save weights in
np.savetxt('/data/wiay/federico/PhD/cosmoflow/COSMOFlow/magnitude_threshold_maps/luminosity_weighted_map/NSIDE_'+str(NSIDE)+'_Luminosity_weight_IN_map_GLADE_'+band+'_zmax'+str(zmax)+'.txt', np.array(w_in_map))
hp.mollview(np.array(w_in_map), rot = (180,0,0), title = 'Weights in catalog', min=0, max=0.007)
hp.graticule(True)
plt.savefig('/data/wiay/federico/PhD/cosmoflow/COSMOFlow/magnitude_threshold_maps/luminosity_weighted_map/NSIDE_Luminosity_weight_IN_map_GLADE_'+band+'_zmax'+str(zmax)+'.png')
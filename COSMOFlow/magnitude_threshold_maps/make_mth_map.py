import pandas as pd
import multiprocessing 
import healpy as hp
from tqdm import tqdm 
import numpy as np
import matplotlib.pyplot as plt

band = 'K'

NSIDE = 32
Npix = hp.nside2npix(NSIDE)

def mth_per_pixel(pix):
    loaded_pix = pd.read_csv('/data/wiay/federico/PhD/cosmoflow/COSMOFlow/pixelated_catalogs/GLADE+_pix/pixel_{}'.format(pix))
    Npix = len(loaded_pix)
    magnitudes = loaded_pix['m'+band].dropna()
    if len(magnitudes) >= 10: 
        mag = np.median(magnitudes)
    else: 
        mag = 0
    
    return mag

with multiprocessing.Pool(10) as p:
    map_mth = list(tqdm(p.imap(mth_per_pixel,np.arange(Npix)), total = Npix))
    
np.savetxt('/data/wiay/federico/PhD/cosmoflow/COSMOFlow/magnitude_threshold_maps/NSIDE_'+str(NSIDE)+'_mth_map_GLADE_'+band+'.txt', map_mth)
    
    
map_mth = np.array(map_mth)

inx_low = np.where(map_mth < 12.5)
    
map_mth[inx_low] = np.nan
    
hp.mollview(np.array(map_mth), title = 'NSIDE = '+str(NSIDE)+', magnitude Threshold map, BAND = '+band ,rot = (180,0,0), cbar = True, cmap = 'plasma')
hp.graticule(True)



# fig = plt.gcf()
# ax = plt.gca()
# image = ax.get_images()[0]
# cmap = fig.colorbar(image, ax=ax, orientation ='horizontal', cmap = 'cividis')

# fig = plt.gcf()
# ax = plt.gca()
# image = ax.get_images()[0]
# cmap = fig.colorbar(image, ax=ax)
plt.savefig('/data/wiay/federico/PhD/cosmoflow/COSMOFlow/magnitude_threshold_maps/NSIDE_32_mth_map_GLADE_'+band+'.png')
plt.show()


    
    


        
        
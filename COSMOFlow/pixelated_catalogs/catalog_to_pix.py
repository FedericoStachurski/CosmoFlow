import h5py 
import pandas as pd
import numpy as np
import healpy as hp
from tqdm import tqdm
import multiprocessing



fn = '/data/wiay/galaxy_catalogs/GLADE+/GLADE+.txt'
chunksize = int(10**6)
df = pd.read_csv(fn, header=None, delim_whitespace=True, 
                 usecols=[8, 9, 18, 25, 27, 31], chunksize = chunksize)
cat = pd.DataFrame()
for chunk in tqdm(df):
    cat = pd.concat([cat, chunk])
    
cat.columns = ['RA', 'dec', 'mK', 'mBj', 'z', 'sigmaz' ]
  



def pix_from_RAdec(NSIDE, RA, dec):
    phi = np.array(np.deg2rad(RA))
    theta = np.pi/2 - np.array(np.deg2rad(dec))
    pix_inx = hp.ang2pix(NSIDE, theta, phi)
    return pix_inx

def get_k_correction(band, z):
    "Apply K-correction"
    #https://arxiv.org/pdf/1709.08316.pdf
    if band == 'mW1':
        k_corr = -1*(4.44e-2+2.67*z+1.33*(z**2.)-1.59*(z**3.)) #From Maciej email
        return k_corr
    elif band == 'mK':
         # https://iopscience.iop.org/article/10.1086/322488/pdf 4th page lhs
        return -6.0*np.log10(1+z)
    elif band == 'mBj':
        # Fig 5 caption from https://arxiv.org/pdf/astro-ph/0111011.pdf
        # Note that these corrections also includes evolution corrections
        return (2.2*z+6*np.power(z,2.))/(1+15*np.power(z,3.))

#K-band_correction
indx_K_less = np.where(cat.z <= 0.25)[0]
cat['mK'].loc[indx_K_less] = cat['mK'].loc[indx_K_less] - get_k_correction('mK', np.array(cat['z'].loc[indx_K_less]))
# cat['mK'].loc[indx_K_more] = cat['mK'].loc[indx_K_more] - get_k_correction('mK', np.array(cat['z'].loc[indx_K_more]))

#Bj-band_correction
cat['mBj'] = cat['mBj'] - get_k_correction('mBj', np.array(cat['z']))



NSIDE = 128
print(NSIDE)
Npix = hp.nside2npix(NSIDE)


pixel_id = pix_from_RAdec(NSIDE, cat.RA, cat.dec)
cat['pix'] = pixel_id


def save_pixels(pix):
    indicies_temp = np.where(np.array(cat['pix']) == pix)
    
    pix_cat = cat.iloc[indicies_temp]
    pix_cat.to_csv('GLADE+_pix_NSIDE_{}/pixel_{}'.format(NSIDE,pix), index = False )
    #np.save('GLADE+_pix/pixel_{}'.format(pix), pix_cat)


with multiprocessing.Pool(10) as p:
    list(tqdm(p.imap(save_pixels,np.arange(Npix)), total = Npix)) 















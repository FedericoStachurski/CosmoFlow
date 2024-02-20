import numpy as np
import pandas as pd
import multiprocessing
import healpy as hp
from tqdm import tqdm 


NSIDE = 32  #Define NSIDE for healpix map
Npix = hp.nside2npix(NSIDE)
threads = 2

def load_cat_by_pix(pix): #load pixelated catalog 
    loaded_pix = pd.read_csv('../pixelated_catalogs/GLADE+_pix/pixel_{}'.format(pix)) #Include NSIDE in the name of folders 
    return loaded_pix

def load_pixel(pix, catalog_pixelated): #load pixel from catalog
    loaded_pix = catalog_pixelated[pix]
    return loaded_pix, len(loaded_pix)


with multiprocessing.Pool(threads) as p: #begin multiprocesseing for loading the catalog
    catalog_pixelated = list(tqdm(p.imap(load_cat_by_pix,np.arange(Npix)), total = Npix, desc = 'GLADE+ catalog, NSIDE = {}'.format(NSIDE)))


def select_gal_from_pix(pixels_H0_gamma_para): 
    # "Selects galaxies from pixel using pixel index and associated H0 to pixel"
    # "Input: tuple(pixel_inx,H0); 
    # "Returns: dataframe of pixel id (z, ra, dec...)" 
    pixel, H0 = pixels_H0_gamma_para
    loaded_pixel, Ngalpix = load_pixel(int(pixel), catalog_pixelated)
    loaded_pixel = loaded_pixel[['z','RA','dec', 'sigmaz', 'm'+band]] #load pixel 
    loaded_pixel = loaded_pixel.dropna() # drop any Nan values 
    
    loaded_pixel = loaded_pixel[loaded_pixel.z >= zmin] #check if z is greater than zmin 
    loaded_pixel['RA'] = np.deg2rad(loaded_pixel['RA']) #convert RA and dec into radians 
    loaded_pixel['dec'] = np.deg2rad(loaded_pixel['dec'])

    Ngalpix = len(loaded_pixel) #get number of galaxies in pixel 
    
    if loaded_pixel.empty is False: #if there are galaxies in the pixel
        z_gal_selected = loaded_pixel.z #get redshift 
        repeated_H0_in_pix = np.ones(Ngalpix)*H0 #for that specific pixel, make vector of H0s used for the specific pixel 
        dl_galaxies = cosmology.fast_z_to_dl_v2(np.array(z_gal_selected).flatten(),np.array(repeated_H0_in_pix).flatten()) #compute distances of galaxies using redshift and H0 
        
        #get luminsoity
        absolute_mag = cosmology.abs_M(loaded_pixel['m'+band],dl_galaxies)
        luminosities =  cosmology.mag2lum(absolute_mag)
        
        #weights = L * madau(z) * (1/(1+z))
        weights_gal = luminosities * z_class.time_z(z_gal_selected) #* z_class.Madau_factor(z_gal_selected, zp, gamma, k) 
        weights_gal /= np.sum(weights_gal) # check weights sum to 1
        gal_id = np.random.choice(np.arange(Ngalpix), size = 1, p = weights_gal) #random choice of galaxy in the pixel 
        return loaded_pixel.iloc[gal_id,:]
    
    else: #if no galaxies in pixel, return None Total for federico: 1 jobs; 0 completed, 0 removed, 1 idle
        return None 
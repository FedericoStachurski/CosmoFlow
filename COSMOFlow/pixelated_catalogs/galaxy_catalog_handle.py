import h5py 
import pandas as pd
import numpy as np
import healpy as hp
from tqdm import tqdm
import multiprocessing
import os

class GladePlusCatalog:
    def __init__(self, file_path, nside=128, chunksize=int(1e6), Pool = 10):
        # /mnt/zfshome4/cbc.cosmology/MDC/Catalogs/
        """
        Initializes the GladePlusCatalog class.
        
        Parameters:
        - file_path (str): The file path to the GLADE+ data.
        - nside (int): Healpix NSIDE value.
        - chunksize (int): The size of chunks to read the data in.
        """
        self.file_path = file_path
        self.nside = nside
        self.chunksize = chunksize
        self.data = pd.DataFrame()
        self.npix = hp.nside2npix(self.nside)
        self.Pool = Pool

    def load_data(self):
        """
        Load the GLADE+ data from the specified file path in chunks and concatenate them.
        """
        df_chunks = pd.read_csv(self.file_path, header=None, sep='\s+', 
                               usecols=[8, 9, 18, 25, 27, 31], chunksize=self.chunksize)
        for chunk in tqdm(df_chunks, desc='Loading GLADE+ Data'):
            self.data = pd.concat([self.data, chunk])
        
        self.data.columns = ['RA', 'dec', 'mK', 'mBj', 'z', 'sigmaz']

    @staticmethod
    def pix_from_RAdec(nside, ra, dec):
        """
        Calculate Healpix pixel indices from RA and Dec values.
        
        Parameters:
        - nside (int): Healpix NSIDE value.
        - ra (array-like): Right Ascension values.
        - dec (array-like): Declination values.
        
        Returns:
        - np.array: Array of pixel indices.
        """
        phi = np.deg2rad(ra)
        theta = np.pi / 2 - np.deg2rad(dec)
        return hp.ang2pix(nside, theta, phi)

    @staticmethod
    def get_k_correction(band, z):
        """
        Apply K-correction based on the band and redshift values.
        
        Parameters:
        - band (str): The band name ('mW1', 'mK', or 'mBj').
        - z (float or np.array): Redshift value(s).
        
        Returns:
        - float or np.array: K-correction value(s).
        """
        if band == 'mW1': #### Need to add papers
            return -1 * (4.44e-2 + 2.67 * z + 1.33 * (z ** 2) - 1.59 * (z ** 3))
        elif band == 'mK':
            return -6.0 * np.log10(1 + z)
        elif band == 'mBj':
            return (2.2 * z + 6 * np.power(z, 2)) / (1 + 15 * np.power(z, 3))
        else:
            raise ValueError(f"Unknown band: {band}")

    def apply_k_corrections(self):
        """
        Apply K-corrections to the 'mK' and 'mBj' bands.
        """
        indx_K_less = self.data['z'] <= 0.25
        self.data.loc[indx_K_less, 'mK'] -= self.get_k_correction('mK', self.data.loc[indx_K_less, 'z'])
        self.data['mBj'] -= self.get_k_correction('mBj', self.data['z'])

    def assign_pixels(self):
        """
        Assign Healpix pixel indices to the catalog data.
        """
        self.data['pix'] = self.pix_from_RAdec(self.nside, self.data['RA'], self.data['dec'])

    def save_pixels(self, output_dir):
        """
        Save data for each pixel into individual CSV files in the specified output directory.
        If the directory does not exist, it will be created.
        If the directory already exists, the user will be warned.
        
        Parameters:
        - output_dir (str): The directory where pixel files will be saved.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            print(f"Directory {output_dir} already exists. Please choose a different directory or delete the existing one.")
            return

        with multiprocessing.Pool(self.Pool) as pool:
            list(tqdm(pool.imap(self.save_pixel_data, [(pix, output_dir) for pix in np.arange(self.npix)]), total=self.npix, desc='Saving Pixels'))

    def save_pixel_data(self, args):
        """
        Save data for a single pixel into a CSV file.
        
        Parameters:
        - args (tuple): Tuple containing the pixel index and the output directory.
        """
        pix, output_dir = args
        indicies_temp = np.where(self.data['pix'] == pix)[0]
        pix_cat = self.data.iloc[indicies_temp]
        pix_cat.to_csv(os.path.join(output_dir, f'pixel_{pix}.csv'), index=False)


from cosmology_functions import cosmology 
from cosmology_functions import priors 
import numpy as np
import cupy as xp

  
class RedshiftGW_fast_z_para(object):
    def __init__(self, parameters, run, zmin,zmax, SNRth):
        self.parameters = parameters
        self.run = run
        self.z_grid = np.linspace(zmin,zmax,250)
        self.dz = np.diff(self.z_grid)[0]
        self.SNRth = SNRth
        
        if self.run == 'O3':
            self.magic_snr_dl = 71404.52441406266
        elif self.run == 'O2':
            self.magic_snr_dl = 62386.692042968854
        elif self.run == 'O1':
            self.magic_snr_dl = 48229.26957734367

    def zmax_H0(self,H0, SNRth):
        return cosmology.fast_dl_to_z_v2(self.magic_snr_dl/SNRth,H0)
    
    
    def Madau_factor(self, z, zp, gamma, k):
        num = (1+(1+zp)**(-gamma-k))*(1+z)**(gamma)
        den = 1 + ((1+z)/(1+zp))**(gamma+k)
        return num/den

    def p_sz(self,z, lam = 3):
        return (1+z)**(self.parameters['lam'])

    def time_z(self,z):
        return 1/(1+z)
    

    def p_z_zmax(self,z,  zp, gamma, k, H0, Om0, w0, SNRth):
        zmax = self.zmax_H0(H0, SNRth)
        priorz = self.Madau_factor(z, zp, gamma, k)  * priors.p_z_omega_EoS(z, Om0, w0) * self.time_z(z) 

        if np.size(z) > 1: 
            inx_0 = np.where(z > zmax)[0]
            priorz[inx_0] = 0.00
            return priorz
        else: 
            if z > zmax:
                priorsz = 0
                return priorz

    def make_cdfs(self, parameters):   
        zp, gamma, k, H0 = parameters  
        pdf = self.p_z_zmax(self.z_grid, zp, gamma, k, H0, 0.3, 0.0, self.SNRth)
        cdf = np.cumsum(pdf*self.dz)
        cdf /= np.max(cdf)
        return cdf

    def draw_z_zmax(self, Nsamples,cdfs):
        N = len(cdfs)
        cdfs_snake = xp.asarray(np.concatenate(cdfs)) 
        zlist = np.ndarray.tolist(self.z_grid)
        zlist = N*zlist
        z_array = xp.asarray(zlist)
        # print(z_array)
        cdfs_snake = cdfs_snake + xp.repeat(xp.arange(N), len(self.z_grid))
        t = xp.random.uniform(0,1, size = N*Nsamples) + xp.repeat(xp.arange(N), Nsamples)
        return xp.interp(t, cdfs_snake, z_array).get()

from cosmology_functions import cosmology 
from cosmology_functions import priors 
import numpy as np
import cupy as xp

  
class RedshiftGW_fast_z_para(object):
    def __init__(self, parameters, run, zmin,zmax, SNRth, population ='BBH', fast = True):
        self.parameters = parameters
        self.run = run
        self.z_grid = np.linspace(zmin,zmax,250)
        self.dz = np.diff(self.z_grid)[0]
        self.SNRth = SNRth
        self.population = population
        self.fast = fast 

        if self.run == 'O4':
            self.magic_snr_dl = 120_000# 99.999% quantile
        
        elif self.run == 'O3':
            if self.population == 'NSBH':
                self.magic_snr_dl = 13201.953173828133 #0.995 PERCENTILE
            else: 
                # self.magic_snr_dl = 71404.52441406266 ### TOO small
                # self.magic_snr_dl = 109638.88392302892 * 1.2 # 99.999% quantile
                self.magic_snr_dl = 100_000# 99.999% quantile

        elif self.run == 'O2':
            self.magic_snr_dl = 65004.52441406266 #59645000.44 # 89547.08
        elif self.run == 'O1':
            self.magic_snr_dl = 64859.84 

    def zmax_H0(self,H0, SNRth):
        return cosmology.fast_dl_to_z_v2(self.magic_snr_dl/SNRth,H0)
    
    def zmax_H0_Om0_w0(self,H0, SNRth):
        return cosmology_functions.cosmology.z_to_dl_H_Omegas_EoS(self.magic_snr_dl/SNRth,H0, Om0, w0) 
    
    def Step_mth(self, m,mth):
        if m < mth:
            return 0
        else: 
            return 1
  
    
    
    def Madau_factor(self, z, zp, gamma, k):
        num = (1+(1+zp)**(-gamma-k))*(1+z)**(gamma)
        den = 1 + ((1+z)/(1+zp))**(gamma+k)
        return num/den

    def p_sz(self,z, lam = 3):
        return (1+z)**(self.parameters['lam'])

    def time_z(self,z):
        return 1/(1+z)
    
    def p_z_zmax_single_values(self,z,  zp, gamma, k, H0, Om0, w0, SNRth):
        zmax = self.zmax_H0(H0, SNRth)
        priorz = self.Madau_factor(z, zp, gamma, k)  * priors.p_z_omega_EoS(z, Om0, w0) * self.time_z(z) 
        priorz = np.tile(priorz, (1,np.size(zmax)))
        
        if np.size(z) > 1: 
            priorz[z > zmax] = 0.00
        else: 
            if z > zmax:
                priorsz = 0
        return priorz
    

    def p_z_zmax(self,z,  zp, gamma, k, H0, Om0, w0, SNRth):
        zmax = self.zmax_H0(H0, SNRth)
        priorz = self.Madau_factor(z, zp, gamma, k)  * priors.p_z_omega_EoS(z, Om0, w0) * self.time_z(z) 
        # print(np.shape(priorz))

        if self.fast == True:
            
            if np.size(z) > 1: 
                
                priorz[z > zmax] = 0.00

            else: 
                if z > zmax:
                    priorsz = 0
            return priorz
        else: 
            return priorz
        
    def p_z_single_values(self,z,  zp, gamma, k, H0, Om0, w0, SNRth):
        priorz = self.Madau_factor(z, zp, gamma, k)  * priors.p_z_omega_EoS(z, Om0, w0) * self.time_z(z) 
        return priorz
            
 
            
    
    def make_cdfs(self):
        zp = self.parameters['zp'] ; gamma = self.parameters['gamma'] ; k = self.parameters['k']; H0 = self.parameters['H0']
        w0 = self.parameters['w0']; Om0 = self.parameters['Om0']
        if (np.size(zp) == 1) and (np.size(gamma) == 1) and (np.size(k) == 1) and (np.size(Om0) == 1) and (np.size(w0) == 1) and (np.size(H0) != 1) :
            pdf = self.p_z_zmax_single_values(self.z_grid[:,None],  zp, gamma, k, H0[None,:], Om0, w0, self.SNRth)
        elif (np.size(H0) == 1):
            pdf = self.p_z_zmax(self.z_grid[:,None],  zp, gamma, k, H0, Om0, w0, self.SNRth) 
        elif (np.size(Om0) != 1) and (np.size(w0) != 1):
            pdf = self.p_z_zmax(self.z_grid[:,None],  zp, gamma, k, H0[None,:], Om0[None,:], w0[None,:], self.SNRth)  
        else: 
            pdf = self.p_z_zmax(self.z_grid[:,None],  zp[None,:], gamma[None,:], k[None,:], H0[None,:], Om0[None,:], w0[None,:], self.SNRth)
        cdf = np.cumsum(pdf*self.dz, axis = 0)
        cdf /= np.amax(cdf, axis=0)
        return cdf

    def draw_z_zmax(self, Nsamples,cdfs):
        N = np.shape(cdfs)[1]
        # cdfs_snake = xp.asarray(np.concatenate(cdfs)) 
        cdfs_snake = xp.hstack(cdfs.T) #+np.repeat(np.arange(2), 250)
        zlist = np.ndarray.tolist(self.z_grid)
        zlist = N*zlist
        z_array = xp.asarray(zlist)
        cdfs_snake = cdfs_snake + xp.repeat(xp.arange(N),  np.shape(cdfs)[0])
        # cdfs_snake = cdfs_snake + xp.repeat(xp.arange(N), len(self.z_grid))
        t = xp.random.uniform(0,1, size = N*Nsamples) + xp.repeat(xp.arange(N), Nsamples)
        return xp.interp(t, cdfs_snake, z_array).get()

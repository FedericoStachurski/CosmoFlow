import numpy as np
from scipy.special import gamma, gammaincc
import cupy as xp



class Schechter_function(object):
    def __init__(self, band):
        self.band = band
        #Schechter function fitting parameters for GLADE+ computed at H0 = 100 km/s/Mpc
        if self.band == 'Bj': 
            self.LF_para = {'Mabs_min': -22.00 , 'Mabs_max': -16.50, 'alpha': -1.21, 'Mstar': -19.66, 'phi_star': 1.16e-2}
        elif self.band == 'K':
            self.LF_para = {'Mabs_min': -27.00 , 'Mabs_max': -19, 'alpha': -1.09, 'Mstar': -23.39, 'phi_star': 1.16e-2}
        else: 
            raise ValueError(self.band + 'band is not implemented')


    def M_grid_H0(self, low,high, H0):
        return np.linspace(high  + 5*np.log10(H0/100),low  + 5*np.log10(H0/100), 100)

    def LF(self, M,H0):
        phi_star = self.LF_para['phi_star']; alpha = self.LF_para['alpha']; Mstar = self.LF_para['Mstar']
        # Mstar = Mstar + 5*np.log10(H0/100)
        # phi_star = phi_star * (H0/100)**(3)
        return phi_star * 0.4*np.log(10)*10**(-0.4*(M - Mstar)*(alpha + 1))*np.exp(-10**(-0.4*(M - Mstar)))

    def LF_weight_L(self, M,H0):
        phi_star = self.LF_para['phi_star']; alpha = self.LF_para['alpha']; Mstar = self.LF_para['Mstar']
        # Mstar = Mstar + 5*np.log10(H0/100)
        # phi_star = phi_star * (H0/100)**(3)
        return phi_star * 0.4*np.log(10)*10**(-0.4*(M - Mstar)*(alpha + 2))*np.exp(-10**(-0.4*(M - Mstar)))
    
    def cdf_LF(self, M, H0):
        a = self.LF_para['alpha']; Mstar =self.LF_para['Mstar']; Mtop = self.LF_para['Mabs_min']; Mbottom = self.LF_para['Mabs_max']
        # Mstar = Mstar + 5*np.log10(H0/100)
        # Mtop = Mtop  + 5*np.log10(H0/100)
        # Mbottom = Mbottom  + 5*np.log10(H0/100)
        L_Lstar = np.power(10, -0.4*( M - Mstar ))
        Lmin_Lstar = np.power(10, -0.4*( Mbottom - Mstar ))
        Lmax_Lstar = np.power(10, -0.4*( Mtop - Mstar ))
        result = np.array((gammaincc(a+1, L_Lstar) - gammaincc(a+1, Lmax_Lstar)), dtype= float)
        return result 

    def cdf_LF_weighted_L(self, M, H0):
        a = self.LF_para['alpha']; Mstar =self.LF_para['Mstar']; Mtop = self.LF_para['Mabs_min']; Mbottom = self.LF_para['Mabs_max']
        # Mstar = Mstar + 5*np.log10(H0/100)
        # Mtop = Mtop  + 5*np.log10(H0/100)
        # Mbottom = Mbottom  + 5*np.log10(H0/100)
        L_Lstar = np.power(10, -0.4*( M - Mstar ))
        Lmin_Lstar = np.power(10, -0.4*( Mbottom - Mstar ))
        Lmax_Lstar = np.power(10, -0.4*( Mtop - Mstar ))
        result = np.array((gammaincc(a+2, L_Lstar) - gammaincc(a+2, Lmax_Lstar)), dtype= float)
        return result 
    
    
    def sample_M_from_cdf(self, H0, N = 1):
        M_grid = self.M_grid_H0(self.LF_para['Mabs_min'], self.LF_para['Mabs_max'], H0)
        cdf_M = self.cdf_LF(M_grid[::-1], H0)
        t = xp.random.uniform(0, 1, N)
        samples = xp.interp(t,xp.asarray(cdf_M),xp.asarray(M_grid))
        return samples.get()
    

    def sample_M_from_cdf_weighted(self, H0, N = 1):
        M_grid = self.M_grid_H0(self.LF_para['Mabs_min'], self.LF_para['Mabs_max'], H0)
        cdf_M = self.cdf_LF_weighted_L(M_grid[::-1], H0)
        t = xp.random.uniform(0, 1, N)
        samples = xp.interp(t,xp.asarray(cdf_M/np.max(cdf_M)),xp.asarray(M_grid))
        return samples.get()
    

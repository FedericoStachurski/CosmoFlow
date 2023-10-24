import cupy as xp
import scipy as sp
from cupyx.scipy.special import erf
import numpy as np
from scipy.stats import truncnorm


class MassPrior(object):
    def __init__(self, population_parameters, mmax_total = 200):
        self.parameters = population_parameters
        self.mgrid = 250
        self.mmax_total = mmax_total
        self.population_type = population_parameters['name']
        self.m_vect = xp.linspace(0, self.mmax_total, self.mgrid)
        self.dm = xp.diff(self.m_vect)[0]
        # population_parameters = {'beta': 0.81, 'alpha': 3.78, 'mmin': 4.98 ,
        #                  'mmax': 100, 'mu_g': 32.27, 
        #                  'sigma_g': 3.88, 'lambda_peak': 0.03,
        #                  'delta_m': 4.8, 'name': 'BBH-powerlaw-gaussian'}
        
    
    def gaussian_peak(self, xx, mu, sigma, high, low):
        norm = 2**0.5 / np.pi**0.5 / sigma
        norm /= erf((high - mu) / 2**0.5 / sigma) + erf((mu - low) / 2**0.5 / sigma)  #vector of norms
        prob = xp.exp(-xp.power(xx - mu, 2) / (2 * sigma**2)) # array of dims len(xx) * len(mu)
        prob *= norm
        return prob
    
    def power_law(self, xx, index, high, low):
        norm = (high**(-index+1) - low**(-index+1))/(-index + 1)
        power = xx**(-index)/norm 
        if xp.size(xx) > 1: 
            power[xx > high] = 0.00
            power[xx < low] = 0.0
        elif (xx > high) & (xx < low):
            power = 0 
        return power
    
    def power_law_m2(self, xx, index, high, low):
        #### HIGH is m1 
        norm = (high**(index+1) - low**(index+1))/(index + 1)
        power = xx**(index)/norm 
        if xp.size(xx) > 1: 
            power[xx > high] = 0.00
            power[xx < low] = 0.0
        elif (xx > high) & (xx < low):
            power = 0 
        return power
    
    
    def smooth_factor(self,xx, mmin, smooth_scale):
        mprime = xx - mmin
        smooth_exp =  xp.exp((smooth_scale/mprime)+(smooth_scale/(mprime-smooth_scale)))
        ans = (1 + smooth_exp)**-1
        if xp.size(xx) > 1: 
            ans[xx > mmin + smooth_scale] = 1.0
            ans[xx < mmin] = 0.0
        else:
            if xx >  mmin + smooth_scale:
                ans = 0 
            elif xx < mmin:
                ans = 0 
        return ans
    

    def powerlaw_plus_peak_smooth(self,xx): ##Not normalised 
        factor1 = self.power_law(xx, self.parameters['alpha'], self.parameters['mmax'], self.parameters['mmin'])
        factor2 = self.gaussian_peak(xx, self.parameters['mu_g'], self.parameters['sigma_g'], self.parameters['mmax'], self.parameters['mmin'])
        return (factor1*(1-self.parameters['lambda_peak']) + factor2*(self.parameters['lambda_peak']))*self.smooth_factor(xx,self.parameters['mmin'], self.parameters['delta_m'])
    
    def powerlaw_plus_peak_smooth_vect(self,xx): ##Not normalised 
        factor1 = self.power_law(xx[None,:], self.parameters['alpha'][:,None], self.parameters['mmax'][:,None], self.parameters['mmin'][:,None])
        factor2 = self.gaussian_peak(xx[None,:], self.parameters['mu_g'][:,None], self.parameters['sigma_g'][:,None], self.parameters['mmax'][:,None], self.parameters['mmin'][:,None])
        return (factor1*(1-self.parameters['lambda_peak'][:,None]) + factor2*(self.parameters['lambda_peak'][:,None]))*self.smooth_factor(xx[None,:],self.parameters['mmin'][:,None], self.parameters['delta_m'][:,None])
    
    def powerlaw_smooth_m2_vect(self,xx, m1): ##Not normalised 
        factor1 = self.power_law(xx[None,:], self.parameters['alpha'][:,None], m1[:,None], self.parameters['mmin'][:,None])
        return factor1*self.smooth_factor(xx[None,:],self.parameters['mmin'][:,None], self.parameters['delta_m'][:,None])
    
    def make_cdfs_m2(self, m1):
        pdf = self.powerlaw_smooth_m2_vect(self.m_vect, m1)
        cdf = xp.cumsum(pdf*self.dm, axis = 1)
        # print(xp.amax(cdf, axis = 1))
        cdf_maximum = xp.amax(cdf, axis=1)[:,None]
        return cdf/cdf_maximum
    
    def make_cdfs(self):
        pdf = self.powerlaw_plus_peak_smooth_vect(self.m_vect)
        cdf = xp.cumsum(pdf*self.dm, axis = 1)
        # print(xp.amax(cdf, axis = 1))
        cdf_maximum = xp.amax(cdf, axis=1)[:,None]
        return cdf/cdf_maximum
    
    def draw_m(self, Nsamples,cdfs):
        N = len(cdfs)
        cdfs_snake = xp.asarray(np.concatenate(cdfs)) 
        mlist = xp.ndarray.tolist(self.m_vect)
        mlist = N*mlist
        m_array = xp.asarray(mlist)
        # print(z_array)
        cdfs_snake = cdfs_snake + xp.repeat(xp.arange(N), len(self.m_vect))
        t = xp.random.uniform(0,1, size = N*Nsamples) + xp.repeat(xp.arange(N), Nsamples)
        return xp.interp(t, cdfs_snake, m_array).get()

    
    
    
    
    
    
    
        

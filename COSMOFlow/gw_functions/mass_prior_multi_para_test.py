import cupy as xp
import scipy as sp
from cupyx.scipy.special import erf
import numpy as np
from scipy.stats import truncnorm


class MassPrior(object):
    def __init__(self, population_parameters, mmax_total = 200, mgrid = 200):
        self.parameters = population_parameters
        self.mgrid = mgrid
        self.mmax_total = mmax_total
        self.population_type = population_parameters['name']
        self.m_vect = xp.linspace(0, self.mmax_total, self.mgrid)
        
        self.log_m_vect = xp.exp(xp.linspace(xp.log(1), xp.log(self.mmax_total), self.mgrid))
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
        
        if xp.size(xx) > 1: 
            prob[xx > high] = 0.0
            prob[xx < low] = 0.0
        elif (xx > high) and (xx < low):
            prob = 0 
        
        return prob
    
    def power_law(self, xx, index, high, low):
        norm = (high**(-index+1) - low**(-index+1))/(-index + 1)
        power = xx**(-index)/norm 
        if xp.size(xx) > 1: 
            power[xx > high] = 0.0
            power[xx < low] = 0.0
        elif (xx > high) and (xx < low):
            power = 0 
        return power
    
    # def power_law_m2(self, xx, index, high, low):
    #     #### HIGH is m1 
    #     norm = (high**(index+1) - low**(index+1))/(index + 1)
    #     power = xx**(index)/norm 
    #     if xp.size(xx) > 1: 
    #         power[xx > high] = 0.00
    #         power[xx < low] = 0.0
    #     elif (xx > high) & (xx < low):
    #         power = 0 
    #     return power
    
    
    def smooth_factor(self,xx, mmin, smooth_scale):
        mprime = xx - mmin
        smooth_exp =  xp.exp((smooth_scale/mprime)+(smooth_scale/(mprime-smooth_scale)))
        ans = (1 + smooth_exp)**-1
        if xp.size(xx) > 1: 
            ans[xx > mmin + smooth_scale] = 1.0
            ans[xx < mmin] = 0.0
        else:
            if xx >  mmin + smooth_scale:
                ans = 1
            elif xx < mmin:
                ans = 0 
        return ans
    

    def powerlaw_plus_peak_smooth(self,xx): ##Not normalised 
        factor1 = self.power_law(xx, self.parameters['alpha'], self.parameters['mmax'], self.parameters['mmin'])
        factor2 = self.gaussian_peak(xx, self.parameters['mu_g'], self.parameters['sigma_g'], self.parameters['mmax'], self.parameters['mmin'])
        return (factor1*(1-self.parameters['lambda_peak']) + factor2*(self.parameters['lambda_peak']))*self.smooth_factor(xx,self.parameters['mmin'], self.parameters['delta_m'])
    
    def powerlaw_plus_peak_smooth_2gaussian(self,xx): ##Not normalised 
        factor1 = self.power_law(xx, self.parameters['alpha'], self.parameters['mmax'], self.parameters['mmin'])
        factor2 = self.gaussian_peak(xx, self.parameters['mu_g'], self.parameters['sigma_g'], self.parameters['mmax'], self.parameters['mmin'])
        factor3 = self.gaussian_peak(xx, self.parameters['mu_g2'], self.parameters['sigma_g2'], self.parameters['mmax'], self.parameters['mmin'])
        return (factor1*(1-self.parameters['lambda_peak']) + (self.parameters['lambda_peak'])*(factor2*(1-self.parameters['lambda_peak2']) + factor3*self.parameters['lambda_peak2']))*self.smooth_factor(xx,self.parameters['mmin'], self.parameters['delta_m'])
    
    
    def powerlaw_plus_peak_smooth_vect(self,xx): ##Not normalised 
        factor1 = self.power_law(xx[None,:], self.parameters['alpha'][:,None], self.parameters['mmax'][:,None], self.parameters['mmin'][:,None])
        factor2 = self.gaussian_peak(xx[None,:], self.parameters['mu_g'][:,None], self.parameters['sigma_g'][:,None], self.parameters['mmax'][:,None], self.parameters['mmin'][:,None])
        return (factor1*(1-self.parameters['lambda_peak'][:,None]) + factor2*(self.parameters['lambda_peak'][:,None]))*self.smooth_factor(xx[None,:],self.parameters['mmin'][:,None], self.parameters['delta_m'][:,None])
    
    def powerlaw_smooth_m2(self,xx, m1, beta = None , mmin = None, delta_m = None ): ##Not normalised
        if beta is None:
            beta = self.parameters['beta']
            mmin = self.parameters['mmin']
            delta_m = self.parameters['delta_m']
        else: 
            beta = xp.array(beta)
            mmin = xp.array(mmin)
            delta_m = xp.array(delta_m)
        
        # print(type(m1))
        
        factor1 = self.power_law(xx, beta, m1, mmin)
        # print(self.smooth_factor(xx,mmin,delta_m))
        return factor1*self.smooth_factor(xx,mmin,delta_m)


    def powerlaw_smooth_m2_vect(self,xx, m1, beta = None , mmin = None, delta_m = None ): ##Not normalised
        if beta is None:
            beta = self.parameters['beta'][:,None]
            mmin = self.parameters['mmin'][:,None]
            delta_m = self.parameters['delta_m'][:,None]
        else: 
            beta = xp.array(beta[:,None])
            mmin = xp.array(mmin[:,None])
            delta_m = xp.array(delta_m[:,None])
        
        # print(type(m1))
        
        factor1 = self.power_law(xx, beta, m1[:,None], mmin)
        # print(self.smooth_factor(xx,mmin,delta_m))
        return factor1*self.smooth_factor(xx,mmin,delta_m)
    
    
    def make_cdfs_m2(self, m1, m_array_long = None, prior_pdf = None):
        m_vect_m2 = xp.linspace(0, m1, self.mgrid, axis = 1)
        # dm = (np.diff(m_vect_m2)[:,0]).reshape(len(m1), 1)
        if prior_pdf is None:
            if m_array_long is None:
                pdf = self.powerlaw_smooth_m2_vect(m_vect_m2, m1)
            else: 
                pdf = self.powerlaw_smooth_m2_vect(m_array_long, m1)
        else: 
            pdf = prior_pdf
        cdf = xp.cumsum(pdf, axis = 1)
        cdf_maximum = xp.amax(cdf, axis=1)[:,None]
        return cdf/cdf_maximum
    
    def make_cdfs(self, log_space = False, m1_arr = None):
        if m1_arr is None:
            if log_space is True:
                 pdf = self.powerlaw_plus_peak_smooth_vect(self.log_m_vect)
            else: 
                pdf = self.powerlaw_plus_peak_smooth_vect(self.m_vect)
        else: 
            pdf = self.powerlaw_plus_peak_smooth_vect(m1_arr)
            
        cdf = xp.cumsum(pdf, axis = 1)
        cdf_maximum = xp.amax(cdf, axis=1)[:,None]
        return cdf/cdf_maximum
    
    def draw_m(self, Nsamples,cdfs, m_array_long = None, log_space = False):
        N = len(cdfs)
        # print(N)
        cdfs_snake = xp.asarray(np.concatenate(cdfs)) 
        if m_array_long is not None: 
            mlist = m_array_long
        else: 
            if log_space is True:
                mlist = xp.ndarray.tolist(self.log_m_vect)
            else: 
                mlist = xp.ndarray.tolist(self.m_vect)
            mlist = N*mlist
        m_array = xp.asarray(mlist)
        # print(z_array)
        cdfs_snake = cdfs_snake + xp.repeat(xp.arange(N), self.mgrid)
        t = xp.random.uniform(0,1, size = N*Nsamples) + xp.repeat(xp.arange(N), Nsamples)
  
        # print(np.shape((cdfs_snake)), np.shape(m_array), np.shape(t))
        return xp.interp(t, cdfs_snake, m_array).get()
    
    def draw_m_simple(self, Nsamples,cdf, m_array_long = None):
        t = xp.random.uniform(0,1, size = Nsamples)
        return xp.interp(t, cdf, self.m_vect).get()
    
    
    def m1_m2_sampling(self, N, Nselect, beta, mmin, delta_m, cdfs_m1, mass_class):
        

        samples = mass_class.draw_m(Nselect, cdfs_m1)

        while True:
            inx_nan = (np.where((samples < mmin.get())))[0]
            # print('NaN m1: {}'.format(len(inx_nan)))
            if len(inx_nan) != 0 :
                
                samples[inx_nan] = mass_class.draw_m(1, np.repeat(cdfs_m1,Nselect, axis = 0)[inx_nan, :])
            else:
                break

        m1 = samples

        # m2_arr = xp.array(10**(xp.linspace(xp.log10(mmin-1e-8), xp.log10(xp.array(samples)), mass_class.mgrid))).T
        m2_arr = xp.array((xp.linspace(mmin,xp.array(samples), mass_class.mgrid))).T
        m2_arr_long = (m2_arr.reshape(int(N*Nselect*mass_class.mgrid),-1)).flatten()


        pdfs_m2 = mass_class.powerlaw_smooth_m2_vect(m2_arr, xp.array(samples),
                                                     beta = beta,
                                                     mmin = mmin,
                                                     delta_m = delta_m)

        cdfs_m2 = mass_class.make_cdfs_m2(xp.array(samples),
                                          prior_pdf = pdfs_m2)

        samples_m2 = mass_class.draw_m(1, cdfs_m2,
                                       m_array_long = m2_arr_long)

        while True: 
            # inx_nan = (np.where((np.isnan(samples_m2) == True) and (samples_m2 > samples)))[0]#
            inx_nan = np.where((np.isnan(samples_m2)) | (samples_m2 > samples))[0]
            # print('NaN m2: {}'.format(len(inx_nan)))
            if len(inx_nan) != 0:
                
                m2_arr = xp.linspace((mmin)[inx_nan],xp.array(samples[inx_nan]), mass_class.mgrid).T
                m2_arr_long = (m2_arr.reshape(int(len(inx_nan)*mass_class.mgrid),-1)).flatten()
                pdfs_m2[inx_nan] = mass_class.powerlaw_smooth_m2_vect(m2_arr, xp.array(samples[inx_nan]),
                                                                      beta = beta[inx_nan],
                                                                      mmin = mmin[inx_nan],
                                                                      delta_m = delta_m[inx_nan])

                cdfs_m2[inx_nan] = mass_class.make_cdfs_m2(xp.array(samples)[inx_nan], prior_pdf = pdfs_m2[inx_nan])
                samples_m2[inx_nan] = mass_class.draw_m(1, cdfs_m2[inx_nan],  m_array_long = m2_arr_long)
            else:
                break

        m2 = samples_m2
        return m1, m2
    
    # def sample_m1_m2()

    
    
    
    
    
    
    
        

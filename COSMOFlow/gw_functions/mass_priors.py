import pandas as pd
import astropy.units as u
from astropy.coordinates import Distance
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ncx2
from tqdm import tqdm
import bilby
from scipy.stats import norm
from cosmology_functions import cosmology, priors
from scipy.interpolate import splrep, splev
from bilby.core.prior import Uniform, Sine, Constraint, Cosine
from scipy.integrate import  quad
import numpy.random as rn
import sys
import multiprocessing
from scipy import interpolate
from scipy.stats import binom, norm
import multiprocessing
from  gw_functions import gwcosmo_priors
from  gw_functions import mass_distribution_cupy
import cupy as xp 



class MassPrior_sample(object):
    def __init__(self, parameters, name):
         # hyper_params_dict = {'beta': 0.81, 'alpha': 3.78, 'mmin': 4.98 ,'mmax': 100, 'mu_g': 32.27, 'sigma_g': 3.88, 
         #     'lambda_peak': 0.03,'delta_m': 4.8} 
        self.hyper_params_dict = parameters
        self.name = self.hyper_params_dict['name']


    def POWERLAW_PEAK_SMOOTH_CUPY(self, N):
        truths = dict(mmin =self.hyper_params_dict['mmin'], mmax = self.hyper_params_dict['mmax'], 
                  mpp=self.hyper_params_dict['mu_g'], 
                  sigpp=self.hyper_params_dict['sigma_g'], 
                  alpha= self.hyper_params_dict['alpha'], 
                  beta = self.hyper_params_dict['beta'], 
                  lam =self.hyper_params_dict['lambda_peak'], delta_m = self.hyper_params_dict['delta_m'])
        for pr in truths.keys():
            truths[pr] = xp.array(truths[pr],dtype=xp.float64)
        m1, m2 = [], []
        Ncupy = 1e5
        batches_mq = N/Ncupy
        if batches_mq < 1:
            batches_mq = 1

        for _ in range(int(batches_mq)+1):
            m1_samples, m2_samples = mass_distribution_cupy.sample_m1_q_fast(truths, int(Ncupy),m1_grid_size=1000, q_grid_size = 250)
            m1.append(m1_samples.get()) ; m2.append(m2_samples.get()) 

        if batches_mq == 1:
            m1 = np.array(m1[0])[:int(N)] ; m2 = np.array(m2[0])[:int(N)] 
        else: 
            m1 = np.concatenate(m1) ; m2 = np.concatenate(m2) 
            m1 = np.array(m1)[:int(N)] ; m2 = np.array(m2)[:int(nxN)] 
        return m1, m2


    def PL_PEAK_GWCOSMO(self, N):
        # name = 'BBH-powerlaw-gaussian'
        hyper_params = {'beta': self.hyper_params_dict['beta'], 'alpha': self.hyper_params_dict['alpha'],
                             'mmin': self.hyper_params_dict['mmin'] ,'mmax': self.hyper_params_dict['mmax'], 
                             'mu_g': self.hyper_params_dict['mu_g'], 'sigma_g': self.hyper_params_dict['sigma_g'], 
                             'lambda_peak': self.hyper_params_dict['lambda_peak'],'delta_m': self.hyper_params_dict['delta_m']} 
        mass_prior_class = gwcosmo_priors.mass_prior(self.name, hyper_params)
        m1 , m2 = mass_prior_class.sample(N)
        return m1, m2 

# def sample_PL_m1m2(Nsamples, alpha, Mmax = 100, Mmin = 5):
#     def draw_cumulative(N,alpha, distribution):
#         #grid = np.linspace(-23,-5,100)
#         cdf = np.zeros(len(m_vec))
#         for i in range(len(m_vec)):
#             cdf[i] = quad(lambda M: distribution(alpha, M),  m_vec  [0], m_vec  [i])[0]
#         cdf = cdf/np.max(cdf)     
#         t = rn.random(N)
#         samples = np.interp(t,cdf,m_vec )
#         return samples

#     def PL_mass1(alpha,m1):
#         return m1**(-alpha)

#     Norm = quad(lambda m: PL_mass1(alpha, m), Mmin, Mmax)[0]

#     m_vec = np.linspace(Mmin ,Mmax, 25000)

#     def m2_sample_m1(m1):
#         return np.random.uniform(Mmin, m1, size = 1)[0]

#     m1 = draw_cumulative(Nsamples,alpha, PL_mass1)


#     m2_list = []     
#     for m1_sample in m1:

#         m2_list.append(m2_sample_m1(m1_sample))


#     m2 = np.array(m2_list)
#     return m1, m2

# ############# POWER LAW + PEAK ####################
# alpha, Mmin, Mmax = [3.78, 4.98, 112.5]
# mu, std = [ 32.27, 3.88]
# lamb = 0.03
# beta = 0.81
# delta_m = 4.8
# parameters = {'alpha' : alpha,
#               'mu' : mu,
#               'std' : std,
#               'lambda' : lamb,
#               'beta' : beta,
#               'Mmin' : Mmin,
#               'Mmax' : Mmax,
#               'delta_m': delta_m}

# ########### Compute 2D interpolation for p(m2|Mmax = m1, beta) ##########
# N = 10_000
# zz = np.zeros((N,N))
# m1_max = np.linspace(Mmin+0.1, Mmax, N)
# m2 = np.linspace(Mmin, Mmax, N)
# dm2 = np.diff(m2)[0]


# def pl_m2_array(m2_array, m1_max, m_min, beta, delta_m):
#     p = np.zeros(len(m2_array))
#     inx = np.where(m2_array < m1_max)
#     def _s_factor(m, m_min, delta_m):
#         select_window = (m>m_min) & (m<(delta_m+m_min))
#         select_one = m>=(delta_m+m_min)
#         select_zero = m<=m_min

#         effe_prime = np.ones(len(m))
#         mprime = m - m_min
#         # Definethe f function as in Eq. B7 of https://arxiv.org/pdf/2010.14533.pdf
#         effe_prime[select_window] = np.exp(((delta_m/mprime[select_window])+(delta_m/(mprime[select_window]-delta_m))))
#         to_ret = 1./(effe_prime+1)
#         to_ret[select_zero]=0.
#         to_ret[select_one]=1.
#         return to_ret 

#     p[inx] = _s_factor(m2[inx], m_min, delta_m)*m2[inx]**(-beta)            
#     return p



# for j in tqdm(range(N), desc = 'Computing meshgrid of m2 - Mmax  ,for evalutaion of p(m2|Mmax = m1, beta)'):
#     cdf_m2_j = np.array(pl_m2_array(m2, m1_max[j], parameters['Mmin'], parameters['beta'], parameters['delta_m'])).cumsum()
#     zz[:,j] = cdf_m2_j*dm2 / np.max(cdf_m2_j*dm2)
# f = interpolate.RegularGridInterpolator((m2, m1_max), zz)    


# def power_law_peak_smooth(parameters, m1, dm):
#     def pl(m1, a):
#         return m1**(-a)

#     def gauss(m1, mu, std):
#         return norm.pdf(m1, mu, std)

#     def _s_factor(m1, m_min, delta_m):
#         select_window = (m1>m_min) & (m1<(delta_m+m_min))
#         select_one = m1>=(delta_m+m_min)
#         select_zero = m1<=m_min

#         effe_prime = np.ones(len(m1))
#         mprime = m1 - m_min
#         # Definethe f function as in Eq. B7 of https://arxiv.org/pdf/2010.14533.pdf
#         effe_prime[select_window] = np.exp(((delta_m/mprime[select_window])+(delta_m/(mprime[select_window]-delta_m))))
#         to_ret = 1./(effe_prime+1)
#         to_ret[select_zero]=0.
#         to_ret[select_one]=1.
#         return to_ret

#     POWERLAW = pl(m1, parameters['alpha'])
#     POWERLAW /= np.sum(POWERLAW*dm)
#     GAUSS = gauss(m1, parameters['mu'], parameters['std'])
#     GAUSS /= np.sum(GAUSS*dm)
#     PL_GAUSS = (1-parameters['lambda'])*POWERLAW + (parameters['lambda'])*GAUSS

#     return PL_GAUSS*_s_factor(m1, parameters['Mmin'], parameters['delta_m'])

# def sample_m1_POWERLAW_PEAK_SMOOTH(N):
    
#     m1_grid = np.linspace(parameters['Mmin'], parameters['Mmax'], 1000)
#     dm1 =  np.diff((m1_grid))[0]
#     pm1 =  power_law_peak_smooth(parameters, (m1_grid), dm1)
#     cdf_m1 = (dm1*pm1.cumsum()) / np.max(dm1*pm1.cumsum())
#     t = rn.random(N)    
#     m1_samples = np.interp(t,cdf_m1,m1_grid)   
#     return m1_samples
    


# def sample_m1m2_POWERLAW_PEAK(N):

#     def m_PL_sample(CDF,alpha = parameters['alpha'], Mmax = parameters['Mmax'], Mmin =  parameters['Mmin']):
#         return (((Mmax)**(-alpha+1))*CDF + (1-CDF)*((Mmin)**(-alpha + 1 )))**(1/(-alpha + 1))

#     def m_GAUSS_sample(n, mu = parameters['mu'], std = parameters['std']):
#         return norm.rvs(size=n , loc = mu, scale = std)



#     r = binom.rvs(N, parameters['lambda'], size=1)
#     CDF_PL_m1 = np.random.uniform(0, 1, size = N - r)
#     CDF_PL_m2 = np.random.uniform(0, 1, size = N)

#     m1_pl = m_PL_sample(CDF_PL_m1) ; m1_gauss = m_GAUSS_sample(r)
#     m1 = np.array(list(m1_pl) + list(m1_gauss))
#     m2 = m_PL_sample(CDF_PL_m2, alpha = -parameters['beta'], Mmax = m1)
#     return m1, m2




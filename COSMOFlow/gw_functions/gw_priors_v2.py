
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
np.random.seed(10)


bilby.core.utils.log.setup_logger(log_level=0)


#GW prior para
dl_dist = bilby.gw.prior.UniformComovingVolume(name='luminosity_distance',minimum=5, maximum=20_000)
m1_dist = bilby.core.prior.Uniform(name='mass_1',minimum=2, maximum=150)
m2_dist = bilby.core.prior.Uniform(name='mass_2', minimum=2, maximum=150)
RA_dist = bilby.core.prior.Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic')
dec_dist = bilby.core.prior.Cosine(name='dec', boundary = 'periodic')


a_1_dist = Uniform(name='a_1', minimum=0, maximum=0.99)
a_2_dist = Uniform(name='a_2', minimum=0, maximum=0.99)


tilt_1_dist = Sine(name='tilt_1')
tilt_2_dist = Sine(name='tilt_2')

phi_12_dist = Uniform(name='phi_12', minimum=0, maximum=2 * np.pi, boundary='periodic')
phi_jl_dist = Uniform(name='phi_jl', minimum=0, maximum=2 * np.pi, boundary='periodic')


theta_jn_dist = Sine(name='theta_jn')
psi_dist = Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic')
phase_dist = Uniform(name='phase', minimum=0, maximum=2 * np.pi, boundary='periodic')
geotime_dist = bilby.core.prior.Uniform(name='geo_time', minimum= 1104105616, maximum=1135641616)
                                        
                                        
def sample_PL_m1m2(Nsamples, alpha, Mmax = 100, Mmin = 5):
    def draw_cumulative(N,alpha, distribution):
        #grid = np.linspace(-23,-5,100)
        cdf = np.zeros(len(m_vec))
        for i in range(len(m_vec)):
            cdf[i] = quad(lambda M: distribution(alpha, M),  m_vec  [0], m_vec  [i])[0]
        cdf = cdf/np.max(cdf)     
        t = rn.random(N)
        samples = np.interp(t,cdf,m_vec )
        return samples

    def PL_mass1(alpha,m1):
        return m1**(-alpha)

    Norm = quad(lambda m: PL_mass1(alpha, m), Mmin, Mmax)[0]

    m_vec = np.linspace(Mmin ,Mmax, 25000)

    def m2_sample_m1(m1):
        return np.random.uniform(Mmin, m1, size = 1)[0]

    m1 = draw_cumulative(Nsamples,alpha, PL_mass1)


    m2_list = []     
    for m1_sample in m1:

        m2_list.append(m2_sample_m1(m1_sample))


    m2 = np.array(m2_list)
    return m1, m2

############# POWER LAW + PEAK ####################
alpha, Mmin, Mmax = [3.78, 4.98, 112.5]
mu, std = [ 32.27, 3.88]
lamb = 0.03
beta = 0.81
delta_m = 4.8
parameters = {'alpha' : alpha,
              'mu' : mu,
              'std' : std,
              'lambda' : lamb,
              'beta' : beta,
              'Mmin' : Mmin,
              'Mmax' : Mmax,
              'delta_m': delta_m}

########### Compute 2D interpolation for p(m2|Mmax = m1, beta) ##########
N = 10_000
zz = np.zeros((N,N))
m1_max = np.linspace(Mmin+0.1, Mmax, N)
m2 = np.linspace(Mmin, Mmax, N)
dm2 = np.diff(m2)[0]


def pl_m2_array(m2_array, m1_max, m_min, beta, delta_m):
    p = np.zeros(len(m2_array))
    inx = np.where(m2_array < m1_max)
    def _s_factor(m, m_min, delta_m):
        select_window = (m>m_min) & (m<(delta_m+m_min))
        select_one = m>=(delta_m+m_min)
        select_zero = m<=m_min

        effe_prime = np.ones(len(m))
        mprime = m - m_min
        # Definethe f function as in Eq. B7 of https://arxiv.org/pdf/2010.14533.pdf
        effe_prime[select_window] = np.exp(((delta_m/mprime[select_window])+(delta_m/(mprime[select_window]-delta_m))))
        to_ret = 1./(effe_prime+1)
        to_ret[select_zero]=0.
        to_ret[select_one]=1.
        return to_ret 

    p[inx] = _s_factor(m2[inx], m_min, delta_m)*m2[inx]**(-beta)            
    return p



for j in tqdm(range(N), desc = 'Computing meshgrid of m2 - Mmax  ,for evalutaion of p(m2|Mmax = m1, beta)'):
    cdf_m2_j = np.array(pl_m2_array(m2, m1_max[j], parameters['Mmin'], parameters['beta'], parameters['delta_m'])).cumsum()
    zz[:,j] = cdf_m2_j*dm2 / np.max(cdf_m2_j*dm2)
f = interpolate.RegularGridInterpolator((m2, m1_max), zz)    


def power_law_peak_smooth(parameters, m1, dm):
    def pl(m1, a):
        return m1**(-a)

    def gauss(m1, mu, std):
        return norm.pdf(m1, mu, std)

    def _s_factor(m1, m_min, delta_m):
        select_window = (m1>m_min) & (m1<(delta_m+m_min))
        select_one = m1>=(delta_m+m_min)
        select_zero = m1<=m_min

        effe_prime = np.ones(len(m1))
        mprime = m1 - m_min
        # Definethe f function as in Eq. B7 of https://arxiv.org/pdf/2010.14533.pdf
        effe_prime[select_window] = np.exp(((delta_m/mprime[select_window])+(delta_m/(mprime[select_window]-delta_m))))
        to_ret = 1./(effe_prime+1)
        to_ret[select_zero]=0.
        to_ret[select_one]=1.
        return to_ret

    POWERLAW = pl(m1, parameters['alpha'])
    POWERLAW /= np.sum(POWERLAW*dm)
    GAUSS = gauss(m1, parameters['mu'], parameters['std'])
    GAUSS /= np.sum(GAUSS*dm)
    PL_GAUSS = (1-parameters['lambda'])*POWERLAW + (parameters['lambda'])*GAUSS

    return PL_GAUSS*_s_factor(m1, parameters['Mmin'], parameters['delta_m'])

def sample_m1_POWERLAW_PEAK_SMOOTH(N):
    
    m1_grid = np.linspace(parameters['Mmin'], parameters['Mmax'], 1000)
    dm1 =  np.diff((m1_grid))[0]
    pm1 =  power_law_peak_smooth(parameters, (m1_grid), dm1)
    cdf_m1 = (dm1*pm1.cumsum()) / np.max(dm1*pm1.cumsum())
    t = rn.random(N)    
    m1_samples = np.interp(t,cdf_m1,m1_grid)

#     def m2_m1_multiprocess(i):
#         m2_new =  np.linspace(parameters['Mmin'], m1_samples[i], 1000)
#         cdf_m2_interp = f((m2_new, m1_samples[i]))
#         t = rn.random(1)
#         return np.interp(t,cdf_m2_interp,m2_new)[0]

#     with multiprocessing.Pool(threads) as p:
#         m2_samples = list(tqdm(p.imap(m2_m1_multiprocess,np.arange(N)), total = N, desc = 'Sampling p(m1, m2) = p(m1)p(m2|m1)'))
#     m2_samples = np.array(m2_samples)    
        
        
#     for m1 in tqdm(m1_samples):
#         cdf_m2_interp = f((m2_new, m1))
#         t = rn.random(1)
#         m2_samples.append(np.interp(t,cdf_m2_interp,m2_new)[0])
#     m2_samples = np.array(m2_samples)     
    return m1_samples
    


def sample_m1m2_POWERLAW_PEAK(N):

    def m_PL_sample(CDF,alpha = parameters['alpha'], Mmax = parameters['Mmax'], Mmin =  parameters['Mmin']):
        return (((Mmax)**(-alpha+1))*CDF + (1-CDF)*((Mmin)**(-alpha + 1 )))**(1/(-alpha + 1))

    def m_GAUSS_sample(n, mu = parameters['mu'], std = parameters['std']):
        return norm.rvs(size=n , loc = mu, scale = std)



    r = binom.rvs(N, parameters['lambda'], size=1)
    CDF_PL_m1 = np.random.uniform(0, 1, size = N - r)
    CDF_PL_m2 = np.random.uniform(0, 1, size = N)

    m1_pl = m_PL_sample(CDF_PL_m1) ; m1_gauss = m_GAUSS_sample(r)
    m1 = np.array(list(m1_pl) + list(m1_gauss))
    m2 = m_PL_sample(CDF_PL_m2, alpha = -parameters['beta'], Mmax = m1)
    return m1, m2

def draw_prior(N, distributions, threads = 5):
    "Draw from defined priors in gw_priors.py, does not have distance"
    
    dl = dl_dist.sample(N)
    
    if distributions['mass'] == 'Power-law':
        m1, m2 = sample_PL_m1m2(Nsamples = N, alpha = 2.3, Mmax = 50, Mmin = 5)
        
    if distributions['mass'] == 'PowerLaw+Peak':
        m1, m2 = sample_m1m2_POWERLAW_PEAK(N)
        
    if distributions['mass'] == 'PowerLaw+Peak_SMOOTH':
        m1, m2 = sample_m1m2_POWERLAW_PEAK_SMOOTH(N, threads = threads)    
        
        
        
    elif distributions['mass'] == 'Uniform':
        if N == 1: 
            while True:
                m1 = m1_dist.sample(N)[0] 
                m2 = m2_dist.sample(N)[0] 


                if m1 > m2:
                    m1 = m1 
                    m2 = m2
                else: 
                    temp1 = m1 
                    temp2 = m2

                    m1 = temp2
                    m2 = temp1
                break
        else: 
            m1 = m1_dist.sample(N)
            m2 = m2_dist.sample(N)
            inx = np.where(m1 < m2)[0]
            temp1 = m1[inx]
            temp2 = m2[inx]
            m1[inx] = temp2
            m2[inx] = temp1



    a1 = a_1_dist.sample(N)
    a2 = a_2_dist.sample(N)

    tilt1 = tilt_1_dist.sample(N)
    tilt2 = tilt_2_dist.sample(N)


    RA = RA_dist.sample(N)
    dec = dec_dist.sample(N)

    theta_jn = theta_jn_dist.sample(N)
    phi_jl = phi_jl_dist.sample(N)
    phi_12 = phi_12_dist.sample(N)
    
    psi = psi_dist.sample(N)
    phase = phase_dist.sample(N)
    geotime = geotime_dist.sample(N)

    return  dl, m1, m2, a1, a2, tilt1, tilt2, RA, dec, theta_jn, phi_jl, phi_12, psi, phase, geotime





import pandas as pd
import astropy.units as u
from astropy.coordinates import Distance
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ncx2
from tqdm import tqdm
import bilby

from cosmology_functions import cosmology, priors
from scipy.interpolate import splrep, splev
from bilby.core.prior import Uniform, Sine, Constraint, Cosine
from scipy.integrate import  quad
import numpy.random as rn
import sys
np.random.seed(10)


bilby.core.utils.log.setup_logger(log_level=0)


#GW prior para
dl_dist = bilby.gw.prior.UniformComovingVolume(name='luminosity_distance',minimum=50, maximum=20_000)
m1_dist = bilby.core.prior.Uniform(name='mass_1',minimum=1, maximum=100)
m2_dist = bilby.core.prior.Uniform(name='mass_2', minimum=1, maximum=100)
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




def draw_prior(N):
    "Draw from defined priors in gw_priors.py, does not have distance"
    
    dl = dl_dist.sample(N)
    
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

    return  dl, m1, m2, a1, a2, tilt1, tilt2, RA, dec, theta_jn, phi_jl, phi_12, psi, phase





import numpy as np 
import pandas as pd
import pickle
import time
from scipy.interpolate import splrep, splev
from scipy.integrate import dblquad, quad, tplquad
from tqdm import tqdm
import matplotlib.pyplot as plt
import bilby 
import astropy.constants as const

from cosmology_functions import priors 
from cosmology_functions import cosmology 

from gw_functions import gw_priors
from gw_functions import gw_SNR


N = 50_000

#draw uniform distances
dl = np.random.uniform(10,24000, N)

#draw redshifted masses 
m1z = np.array(np.random.uniform(15,200, N))
m2z = np.array(np.random.uniform(15,200, N))


        
    




#sample GW priors
_, _, a1, a2, tilt1, tilt2, RA, dec, theta_jn = gw_priors.draw_prior()

#compute SNR from injection
rho_obs = gw_SNR.SNR_from_inj( dl, m1*(1+z), m2*(1+z),  a1, a2, tilt1, tilt2, RA, dec, theta_jn, phi_jl, phi_12, psi, phase)





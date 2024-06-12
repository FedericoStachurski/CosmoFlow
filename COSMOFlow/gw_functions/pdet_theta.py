from gw_functions.gw_SNR_v2 import run_bilby_sim
from scipy.stats import ncx2
import bilby
import numpy as np
from astropy import cosmology
import astropy.constants as const
from scipy.integrate import  quad
from scipy.interpolate import splrep, splev
c = const.c.to('km/s').value         #speed of light in km/s


class LikelihoodDenomiantor(object):
    def __init__(self, snr_th, cosmology, ndet):
        self.snr_th = snr_th
        self.cosmology = cosmology
        self.ndet = ndet
        
        
    def p_D_theta(self, SNR):
        p = ncx2.sf(self.snr_th**(2), 2*self.ndet, SNR**2, loc = 0, scale = 1)
        return p


    def p_theta_omega_cosmo(self, theta):

        dl = theta.luminosity_distance
        dec = theta.dec
        theta_jn = theta.theta_jn
        prior_dl = bilby.gw.prior.UniformComovingVolume( minimum=10, maximum=20000, name = 'luminosity_distance',cosmology=self.cosmology) #Check in O3 paper 
        prob_dl = prior_dl.prob(dl)
        prob_dec = np.cos(dec)
        prob_jn = abs(np.sin(theta_jn))

        return prob_dl #* prob_dec * prob_jn

    def p_theta_omega_nocosmo(self, theta):

        dl = theta.luminosity_distance
        dec = theta.dec
        theta_jn = theta.theta_jn
        prob_dl = dl**(2)/(20_000**3 - 10**3)
        prob_dec = np.cos(dec)
        prob_jn = abs(np.sin(theta_jn))

        return prob_dl * prob_dec * prob_jn


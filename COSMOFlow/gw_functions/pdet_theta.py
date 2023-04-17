from gw_functions.gw_SNR_v2 import run_bilby_sim
from scipy.stats import ncx2
import bilby
import numpy as np
from astropy import cosmology
import astropy.constants as const
from scipy.integrate import  quad
from scipy.interpolate import splrep, splev
c = const.c.to('km/s').value         #speed of light in km/s


cosmology=cosmology.FlatLambdaCDM(name="Planck15", H0 = 67.9, Om0 = 0.3065)
zmax = 1.5

def p_D_theta(theta, rth):
#    "Description: Probability of detection conditioned on the GW parameters, p(D|theta) "
#    "Input: list of the parameters"
#    "Output: pdet "


    dl, m1_det, m2_det, a1, a2, tilt1, tilt2, RA, dec, theta_jn, phi_jl, phi_12, psi, geo_time = theta
    #Compute SNR from given parameters
    SNR = run_bilby_sim( dl, m1_det, m2_det, a1, a2, tilt1, tilt2, RA, dec, theta_jn, phi_jl, phi_12, psi, 0, geo_time)
    
    
    #Define pdet by calling the sf of chi2
    def pdet(x):
        p = ncx2.sf(rth**(2), 6, x**2, loc = 0, scale = 1)
        return p
    
    
    return pdet(SNR)


def p_theta_omega(theta, omega_0 = [0.3065, 0, 67.9]):
#    "Description: Probability of GW parameters, p(theta|Omega_0) "
#    "Input: list of the parameters"
#    "Output: p(theta) "
    
    
    omega_m, omega_k, H0 = omega_0
    dl, m1_det, m2_det, a1, a2, tilt1, tilt2, RA, dec, theta_jn, phi_jl, phi_12, psi, phase = theta
    
    #Following the GWTC-2.1 GW transient catalog, the parameters are all uniforms except for dl.
    #the prior on dl is uniform in comoving volume,with flat Lambda-CDM Hubble = 67.90 and Omega_m = 0.3065

    prior_dl = bilby.gw.prior.UniformComovingVolume( minimum=10, maximum=20000, name = 'luminosity_distance',cosmology=cosmology)

    return prior_dl.prob(dl)


@np.vectorize
def z_to_dl_H_Omegas(z,H, omega_m, omega_lambda):
    "distance as a function of z and H0 and density parameters"
    omega_k = 1 - omega_m - omega_lambda
    def E(z):
        return np.sqrt((omega_m*(1+z)**(3) + omega_k*(1+z)**(2) + omega_lambda))
    
    def I(z):
        fact = lambda x: 1/E(x)
        integral = quad(fact, 0, z)
        return integral[0]

    dl  =  (c*(1+z) / H) * I(z)
    return dl

#make look up table objects with splrep and splev
z_dl_grid = np.linspace(0.0001, zmax, 1000)
dl_z_grid = z_to_dl_H_Omegas(z_dl_grid,cosmology.H0.value, cosmology.Om0, 1 - cosmology.Om0)
spl_z_dl = splrep(z_dl_grid, dl_z_grid)
spl_dl_z = splrep(dl_z_grid, z_dl_grid)

@np.vectorize
def fast_dl_to_z(dl):
    spl_dl = spl_dl_z[0]
    spl_z = spl_dl_z[1]
    spl_degree = spl_dl_z[2]
    return splev(dl, (spl_dl,spl_z,spl_degree))





# rth = 8
# dl = 10000

# theta = [dl, 50, 50, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0]

# pD = p_D_theta(theta, rth)
# prior_theta = p_theta_omega(theta)

# print('p_D = {}'.format(pD))
# print('p_theta = {}'.format(prior_theta))

    
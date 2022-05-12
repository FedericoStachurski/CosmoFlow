from gw_SNR import SNR_from_inj
from scipy.stats import ncx2
import bilby
from astropy import cosmology


def p_D_theta(theta, rth):
#    "Description: Probability of detection conditioned on the GW parameters, p(D|theta) "
#    "Input: list of the parameters"
#    "Output: pdet "


    dl, m1_det, m2_det, a1, a2, tilt1, tilt2, RA, dec, theta_jn, phi_jl, phi_12, psi, phase = theta
    #Compute SNR from given parameters
    SNR = SNR_from_inj( dl, m1_det, m2_det, a1, a2, tilt1, tilt2, RA, dec, theta_jn, phi_jl, phi_12, psi, phase)
    
    
    #Define pdet by calling the sf of chi2
    def pdet(x):
        p = ncx2.sf(rth**(2), 4, x**2, loc = 0, scale = 1)
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

    prior_dl = bilby.gw.prior.UniformComovingVolume( minimum=50, maximum=10000, name = 'luminosity_distance',cosmology=cosmology.FlatLambdaCDM(name="Planck15", H0 = 67.9, Om0 = 0.3065))

    return prior_dl.prob(dl)

rth = 8
dl = 10000
theta = [dl, 50, 50, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0]

pD = p_D_theta(theta, rth)
prior_theta = p_theta_omega(theta)

print('p_D = {}'.format(pD))
print('p_theta = {}'.format(prior_theta))

    
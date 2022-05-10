from gw_SNR import SNR_from_inj
from scipy.stats import ncx2


def p_D_theta(theta):
   "Description: Probability of detection conditioned on the GW parameters, p(D|theta) "
   "Input: list of the parameters"
   "Output: pdet "


    dl, m1_det, m2_det, a1, a2, tilt1, tilt2, RA, dec, theta_jn, phi_jl, phi_12, psi, phase = theta
    #Compute SNR from given parameters
    SNR = SNR_from_inj( dl, m1_det, m2_det, a1, a2, tilt1, tilt2, RA, dec, theta_jn, phi_jl, phi_12, psi, phase)
    
    
    #Define pdet by calling the sf of chi2
    def pdet(x):
        return chi2.sf(4, x**2, loc = 0, scale = 1)
    
    
    return pdet(SNR)



    
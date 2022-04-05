import numpy as np
from scipy.interpolate import splrep, splev
from scipy.integrate import dblquad, quad, tplquad
from scipy.interpolate import splrep, splev
from scipy.interpolate import interp2d
import astropy.constants as const
from scipy.stats import norm
from scipy.special import erf

zmax = 10                             # max redshift
H0 = 70                                #Hubble constant
c = const.c.to('km/s').value         #speed of light in km/s
omega_m = 0.3; omega_lambda = 0.7        #cosmological parameters
omega_k = 1- omega_lambda - omega_m

@np.vectorize
def z_to_dl(z,H):
    "distance as a function of z and H0"
    def E(z):
        return np.sqrt((omega_m*(1+z)**(3) + omega_k*(1+z)**(2) + omega_lambda))
    
    def I(z):
        fact = lambda x: 1/E(x)
        integral = quad(fact, 0, z)
        return integral[0]

    dl  =  (c*(1+z) / H) * I(z)
    return dl

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

z_dl_grid = np.linspace(0.01, zmax, 1000)
dl_z_grid = z_to_dl(z_dl_grid,70)
spl_z_dl = splrep(z_dl_grid, dl_z_grid)
spl_dl_z = splrep(dl_z_grid, z_dl_grid)

@np.vectorize
def fast_z_to_dl_v2(z,H):
    spl_z = spl_z_dl[0]
    spl_dl = spl_z_dl[1]*(70/H)
    spl_degree = spl_z_dl[2]
    return splev(z, (spl_z,spl_dl,spl_degree))
@np.vectorize
def fast_dl_to_z_v2(dl,H):
    spl_dl = spl_dl_z[0]*(70/H)
    spl_z = spl_dl_z[1]
    spl_degree = spl_dl_z[2]
    return splev(dl, (spl_dl,spl_z,spl_degree))

def dist_modulus(M,m):
    'Distance Modulus given Absolute Magnitude and '
    result = 10**((m - M - 25) /5)
    return result

@np.vectorize
def z_MmH(M,m,H):
    "redshift in function of absolute magnitude, apparent magnitude and H0"
    result = fast_dl_to_z_v2(dist_modulus(M,m), H)
    return result  



def app_mag(M,dl):
    "apparent magnitude in function of absolute amgnitude and luminosity distance"
    m = M + 5*(np.log10(dl))+25
    return m.flatten()


def abs_M(m,dl):
    "absolute magnitude in function of apparent amgnitude and luminosity distance"
    M = m - 5*(np.log10(dl))-25
    return M


def M_z(chirp_M, z):
    "Source Chirp mass in function of Chirp M and Redshift"
    return chirp_M*(1+z)

def Source_M(M_z,z):
    "Source Mass frame"
    Source_mass = M_z / (1+z)
    return Source_mass 

def snr_Mz_z(redshifted_mass, z, H):
    "SNR in function of redshifted chirp mass, distance, H0"
    dl = fast_z_to_dl_v2(z,H)
    return (2*redshifted_mass/dl) * (250/(2*2.8))

def snr_Mz_z_Omegas(redshifted_mass, z, H, omega_m, omega_lambda):
    "SNR in function of redshifted chirp mass, distance, H0 and omegas"
    dl = z_to_dl_H_Omegas(z,H, omega_m, omega_lambda)
    return (2*redshifted_mass/dl) * (250/(2*2.8))
    
def snr_M_z(source_mass, z, H):
    "SNR in function of redshifted chirp mass, distance, H0"
    dl = fast_z_to_dl_v2(z,H)
    return (2*source_mass*(1+z)/dl) * (250/(2*2.8))



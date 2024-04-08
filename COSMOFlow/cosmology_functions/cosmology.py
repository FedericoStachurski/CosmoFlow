import numpy as np
from scipy.interpolate import splrep, splev
from scipy.integrate import dblquad, quad, tplquad
from scipy.interpolate import splrep, splev
from scipy.interpolate import interp2d
import astropy.constants as const
from scipy.stats import norm
from scipy.special import erf
from astropy.coordinates import spherical_to_cartesian, cartesian_to_spherical
import cupy as xp
import healpy as hp

zmax = 10                             # max redshift
H0 = 70                                #Hubble constant
c = const.c.to('km/s').value         #speed of light in km/s
omega_m = 0.305; omega_lambda = 1 - omega_m        #cosmological parameters
omega_k = 1- omega_lambda - omega_m
L0 = 3.0128e28 #Watts

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


@np.vectorize
def z_to_dl_H_Omegas_EoS(z,H, omega_m, w0):
    "distance as a function of z and H0 and density parameters"
    omega_k = 0 
    omega_lambda = 1 - omega_m #Flat universe
    def E(z):
        return np.sqrt((omega_m*(1+z)**(3) + omega_k*(1+z)**(2) + omega_lambda*(1+z)**(3*(1+w0))))
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
    "redshift in function of absolute magnitude, appythonparent magnitude and H0"
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

def Vc(z_max, H0, Omega_m):
    "Compute comoving volume up to given zmax with H0 and Omega_m parameters"
    Omega_lam = 1 - Omega_m
    h = H0 / 100
    dl = z_to_dl_H_Omegas(z_max, H0, Omega_m, Omega_lam)
    Vc= 4/3*np.pi*dl**(3)

    return Vc
    
def Ngal(Vc, n):
    "Compute Ngal of the universe given Comoving volume "
    N = n*Vc
    return N

def draw_RA_Dec(N):
    #sample RA 
    ra_obs = xp.random.uniform(0,360,N)
    ra_obs = ra_obs* np.pi / 180
    #sample declinations
    P = xp.random.uniform(0,1,N)
    dec = xp.arcsin(2*P-1) 
    return ra_obs.get(), dec.get()

def m_MzH0(M,z,H0):
    return M + 5*(np.log10(fast_z_to_dl_v2(z,H0)))+25


def mag2lum(M):
    return L0*10**(M/(-2.5))

def lum2mag(L):
    return -2.5*np.log10(L/L0)

#transform Polar into cartesian and spins to sigmoids
def spherical_to_cart(dl, ra, dec):
    x,y,z = spherical_to_cartesian(dl, dec, ra)
    return x,y,z

def cart_to_spherical(x,y,z):
    dl, dec, ra = cartesian_to_spherical(x,y,z)
    return dl, ra, dec


def pix_from_RAdec(NSIDE, RA, dec):
    phi = np.array(np.deg2rad(np.rad2deg(RA)))
    theta = np.pi/2 - np.array(np.deg2rad(np.rad2deg(dec)))
    pix_inx = hp.ang2pix(NSIDE, theta, phi)
    return pix_inx


def target_ra_dec(N, pixels_event, NSIDE):
    RA_data, dec_data = [], []
    while True:
        RA, dec = draw_RA_Dec(N)
        pixels_data = pix_from_RAdec(NSIDE, RA, dec)
        indicies = np.where(np.in1d(pixels_data, np.unique(pixels_event)))[0]
        RA_data.append(RA[indicies])
        dec_data.append(dec[indicies])#
        if len(np.concatenate(RA_data)) >= N:
            break
        else:
            continue
    return np.concatenate(RA_data)[:N], np.concatenate(dec_data)[:N]
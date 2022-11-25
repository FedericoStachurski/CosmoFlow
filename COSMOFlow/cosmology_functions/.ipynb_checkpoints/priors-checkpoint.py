import numpy as np
from scipy.interpolate import splrep, splev
from scipy.integrate import dblquad, quad, tplquad

Mmin = 5 ; Mmax = 100 #chirp mass mass boundaries
zmax = 10
omega_m = 0.3; omega_lambda = 0.7    #cosmological parameters
omega_k = 1- omega_lambda - omega_m

@np.vectorize
def p_z(z, omega_m = 0.3):
    "redshift probability distribution function for Omega_M = 0.3 ad Omega_Lambda = 0.7 (not normalized)"
    omega_lambda = 1 - omega_m
    omega_k = 0 
    def E(z):
        return np.sqrt((omega_m*(1+z)**(3) + omega_k*(1+z)**(2) + omega_lambda))
    
    def I(z):
        fact = lambda x: 1/E(x)
        integral = quad(fact, 0, z)
        return integral[0]
    result =  (I(z))**(2) / E(z)
    return  result

#Look up table grids using splrep and splev
z_grid = np.linspace(0,zmax,1000)
pz_grid = np.array([p_z(z) for z in z_grid])
spl_z_pz = splrep(z_grid,pz_grid)

def fast_p_z(z): 
    return splev(z, spl_z_pz)
 

def p_M(M, H, para_dict = None ):
    "Schecter Function (B-band values) (not normalized)"
    if para_dict is None:
        phi = (1.6*10**(-2))*(H/100)**(3)
        alpha = -1.2
        Mc = -21 + 5*np.log10(H/50)
    else: 
        phi = para_dict['phi']
        alpha = para_dict['alpha']
        Mc = para_dict['Mc']
    return (2/5)*phi*(np.log(10))*((10**((2/5)*(Mc-M)))**(alpha+1))*(np.exp(-10**((2/5)*(Mc-M))))   



def p_M_weight_L(M, H, para_dict = None ):
    "Schecter Function (B-band values) (not normalized), luminosity weighted"
    if para_dict is None:
        phi = (1.6*10**(-2))*(H/100)**(3)
        alpha = -1.2
        Mc = -21 + 5*np.log10(H/50)
    else: 
        phi = para_dict['phi']
        alpha = para_dict['alpha']
        Mc = para_dict['Mc']
    return (2/5)*phi*(np.log(10))*((10**((2/5)*(Mc-M)))**(alpha+2))*(np.exp(-10**((2/5)*(Mc-M)))) 


def LF_weight_L_Kband(M,H0, para_dict = None):
    "Schecter Function (K-band values) (normalized), luminosity weighted"
    M = M - 5*np.log10(H0/100)
    h3 = (H0/100)**(3)
    
    if para_dict is None:
        phi_star =1.16*10**(-2)
        alpha = -1.09 + 1
        Mstar = -23.39
    else: 
        phi_star = para_dict['phi']
        alpha = para_dict['alpha']
        Mstar = para_dict['Mc']
    
    
    return phi_star*h3 * 0.4*np.log(10)*10**(-0.4*(M - Mstar)*(alpha + 1))*np.exp(-10**(-0.4*(M - Mstar)))


def LF_weight_L_Bband(M,H0, para_dict = None):
    "Schecter Function (Bj-band values) (normalized), luminosity weighted"
    M = M - 5*np.log10(H0/100)
    h3 = (H0/100)**(3)
    
    if para_dict is None:
        phi_star =1.16*10**(-2)
        alpha = -1.09 + 1
        Mstar = -23.39
    else: 
        phi_star = para_dict['phi']
        alpha = para_dict['alpha']
        Mstar = para_dict['Mc']
    
    
    return phi_star*h3 * 0.4*np.log(10)*10**(-0.4*(M - Mstar)*(alpha + 1))*np.exp(-10**(-0.4*(M - Mstar)))



def ker_p_M(M, H): 
    "Schecter Function (B-band values) (not normalized)"
    phi = (0.002)*(H/50)**(3)
    alpha = -1.2
    Mc = -21 + 5*np.log10(H/50)
    return ((10**((2/5)*(Mc-M)))**(alpha+1))*(np.exp(-10**((2/5)*(Mc-M))))  

def p_Chirp(M):
    "Uniform Chirp mass prior betwee 5 and 100 Msol"
    #print(type(M))
    if isinstance(M,float):
        if M <= 100 and M>= 15:
            return( 1 / (Mmax - Mmin) )
        else:
            return 0 
    if isinstance(M,np.ndarray):  
        mass = np.zeros(len(M))
        
        for i in range(len(M)):
            if M[i] <= 100 and M[i]>= 15:
                mass[i] = ( 1 / (Mmax - Mmin) )
        return np.array(mass).flatten()
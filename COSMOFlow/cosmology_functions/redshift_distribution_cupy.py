from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import priors  
xp = cp

from cupyx.scipy.special import erf
from cupyx.profiler import benchmark
def trapz(y, x=None, dx=1.0, axis=-1):
    y = xp.asanyarray(y)
    if x is None:
        d = dx
    else:
        x = xp.asanyarray(x)
        if x.ndim == 1:
            d = xp.diff(x)
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = xp.diff(x, axis=axis)
    ndim = y.ndim
    slice1 = [slice(None)] * ndim
    slice2 = [slice(None)] * ndim
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    product = d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0
    try:
        ret = product.sum(axis)
    except ValueError:
        ret = xp.add.reduce(product, axis)
    return ret

def trapz_cumsum(y, x=None, dx=1.0, axis=-1, flipped=False):
    y = xp.asanyarray(y)
    if x is None:
        d = dx
    else:
        x = xp.asanyarray(x)
        if x.ndim == 1:
            d = xp.diff(x)
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = xp.diff(x, axis=axis)
    ndim = y.ndim
    slice1 = [slice(None)] * ndim
    slice2 = [slice(None)] * ndim
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    product = d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0
    if not flipped:
        ret = xp.cumsum(product, axis=axis)
    elif flipped:
        ret = xp.cumsum(xp.flip(product, axis=axis), axis=axis)
    return ret



def Madau_factor(z, gamma = 4.59, k = 2.86, zp = 2.47):
    num = (1+(1+zp)**(-gamma-k))*(1+z)**(gamma)
    den = 1 + ((1+z)/(1+zp))**(gamma+k)
    return num/den

def p_sz(z, lam = 3):
    return (1+z)**(lam)

def time_z(z):
    return 1/(1+z)


def p_z(z, parameters):
    gamma,kappa,zp,lam,Om,zmax = [parameters['gamma'],parameters['k'],parameters['zp'],parameters['lam'],parameters['Om'],parameters['zmax']]
    priorz = Madau_factor(z, gamma = gamma, k = kappa, zp = zp)  * priors.p_z(z, omega_m = Om) * time_z(z) * p_sz(z, lam = lam)
    return priorz

def p_z_zmax(z,parameters):
    gamma,kappa,zp,lam,Om,zmax = [parameters['gamma'],parameters['k'],parameters['zp'],parameters['lam'],parameters['Om'],parameters['zmax']]
    priorz = Madau_factor(z, gamma = gamma, k = kappa, zp = zp)  * priors.p_z(z, omega_m = Om) * time_z(z) * p_sz(z, lam = lam)
    
    if np.size(z) > 1: 
        inx_0 = np.where(z > zmax)[0]
        priorz[inx_0] = 0.00
        return priorz
    else: 
        if z > zmax:
            priorsz = 0
            return priorz
        
    
    
    
    
    
    
# def sample_z_zmax_fast(parameters, size, z_grid_size=1000, zmax_grid_size = 250):
#     zvec = xp.linspace(parameters['zmin'],parameters['zmax'], z_grid_size)# (m1_grid_size,)
#     probs = p_z(zvec, parameters)
#     cdf = xp.cumsum(probs)# (m1_grid_size, )
#     cdf /= cdf.max()
#     zsamples = xp.interp(xp.random.random(size), cdf, zvec)# (size, )

#     zmax_vec = xp.linspace(parameters['zmin'], parameters['zmax'], zmax_grid_size)# (q_grid_size, )
#     q_curves = pq(qvec, m1samples, truths) # (q_grid_size, size)
#     cdfs = xp.nan_to_num(xp.cumsum(q_curves, axis=0) / xp.sum(q_curves,axis=0))# (q_grid_size, size)
#     arange = xp.arange(size)
#     cdf_snake = (cdfs + arange[None,:]).T.flatten()# (q_grid_size * size)
#     sample_snake = xp.random.random(size) + arange
#     q_all = (qvec[:,None] + arange[None,:]).T.flatten()# (q_grid_size * size)
#     qsamples = xp.interp(sample_snake, cdf_snake, q_all) - arange# (size)

#     return m1samples, qsamples*m1samples    
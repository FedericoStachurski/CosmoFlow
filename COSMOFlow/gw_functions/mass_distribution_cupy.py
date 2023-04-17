from functools import partial
import numpy as np
from scipy.integrate import quad
from scipy.constants import c
try:
    import cupy as cp
    xp = cp
    from cupyx.scipy.ndimage import map_coordinates
    from cupyx.scipy.special import erf
except ImportError:
    xp = np
    from scipy.ndimage import map_coordinates
    from scipy.special import erf

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

def interp2d(x, xbounds, xsize, grid):
    xin = xp.tile((x - xbounds[0]) / (xbounds[1] - xbounds[0]) * (xsize-1), grid.shape[0])
    yin = xp.arange(grid.shape[0]).repeat(x.size)  # (0,0,...,1,1,...,nhypers-1,nhypers-1,....) 
    coords_in = xp.vstack((yin, xin))
    return map_coordinates(grid, coords_in, order=1).reshape(grid.shape[0],x.size)

def dL(z):
    '''
    Luminosity distance from redshift assuming default cosmology.
    :param z: Redshift
    :return: dL, in Gpc.
    '''
    h = 0.6774
    omega_m = 0.3089
    omega_lambda = 1 - omega_m
    dH = 1e-5 * c / h

    def E(z):
        return np.sqrt((omega_m * (1 + z) ** (3) + omega_lambda))
    def I(z):
        fact = lambda x: 1 / E(x)
        integral = quad(fact, 0, z)
        return integral[0]

    return (1+z) * dH * I(z) * 1e-3

def truncnorm(xx, mu, sigma, high, low):
    # breakpoint()
    x_op = xp.repeat(xx, mu.size).reshape((xx.size,mu.size)).T
    hi_op = xp.repeat(high, xx.size).reshape((high.size,xx.size))
    lo_op = xp.repeat(low, xx.size).reshape((low.size,xx.size))

    norm = 2**0.5 / np.pi**0.5 / sigma
    norm /= erf((high - mu) / 2**0.5 / sigma) + erf((mu - low) / 2**0.5 / sigma)  #vector of norms
    try:
        prob = xp.exp(-xp.power(xx[None,:] - mu[:,None], 2) / (2 * sigma[:,None]**2)) # array of dims len(xx) * len(mu)
        prob *= norm[:,None]  # should be fine considering dimensionality
        prob[x_op < lo_op] = 0
        prob[x_op > hi_op] = 0
    except IndexError:
        prob = xp.exp(-xp.power(xx - mu, 2) / (2 * sigma**2)) # vector of len(xx)
        prob *= norm
        prob *= (xx <= high) & (xx >= low)
    return prob

def powerlaw(xx, lam, xmin, xmax):
    x_op = xp.repeat(xx, lam.size).reshape((xx.size,lam.size)).T
    hi_op = xp.repeat(xmax, xx.size).reshape((xmax.size,xx.size))
    lo_op = xp.repeat(xmin, xx.size).reshape((xmin.size,xx.size))

    norm = (1+lam)/(xmax**(1+lam) - xmin**(1+lam)) # vector of norms
    try:
        out =  xx[None,:]**lam[:,None] * norm[:,None] # array of dims len(xx) * len(lam)
        out[x_op < lo_op] = 0
        out[x_op > hi_op] = 0
    except IndexError:
        out =  xx**lam * norm # array of dims len(xx) * len(lam)
        out *= (xx <= xmax) & (xx >= xmin)
    return out

def smooth_exp(m, smooth_scale):
    return xp.exp((smooth_scale/m)+(smooth_scale/(m-smooth_scale)))

def smoothing(masses, mmin, delta_m):
    big_masses = xp.repeat(masses, delta_m.size).reshape((masses.size, delta_m.size)).T
    lows = xp.repeat(mmin, masses.size).reshape((mmin.size,masses.size))
    deltas = xp.repeat(delta_m, masses.size).reshape((delta_m.size,masses.size))
    try:
        ans = (1 + smooth_exp(masses[None,:] - mmin[:,None], delta_m[:,None]))**-1
        ans[big_masses < lows] = 0
        ans[big_masses > lows + deltas] = 1
    except IndexError:
        ans = (1 + smooth_exp(masses - mmin, delta_m))**-1
        ans[masses < mmin] = 0
        ans[masses > mmin + delta_m] = 1
    return ans

def ligo_ppop(m, parameters):
    tcs = two_component_single(m, parameters['alpha'],parameters['mmin'],parameters['mmax'],parameters['lam'],parameters['mpp'],parameters['sigpp'])
    smth = smoothing(m, parameters['mmin'], parameters['delta_m'])
    return tcs * smth  # NOT normalised!

def two_component_single(
    mass, alpha, mmin, mmax, lam, mpp, sigpp, gaussian_mass_maximum=100
):
    p_pow = powerlaw(mass, -alpha, mmin, mmax)  # 2d array, dims N_samp * N_hp
    p_norm = truncnorm(mass, mu=mpp, sigma=sigpp, high=xp.asarray(gaussian_mass_maximum), low=mmin)  # same as above

    try:
        prob = (1 - lam[:,None]) * p_pow + lam[:,None] * p_norm
    except IndexError:
        prob = (1 - lam) * p_pow + lam * p_norm
    return prob

def p1(m, p1norm, truths):
    p = ligo_ppop(m, truths)
    return p/p1norm
    
def p2_no_m1(m, beta, mmin, delta_m):
    return m**beta * smoothing(m, mmin, delta_m)

def p2_total(m, Kfunc, p1norm, beta, mmin, delta_m):
    return p2_no_m1(m, beta, mmin, delta_m)/p1norm * Kfunc(m) # integral is a function of m

def construct_I(mvec, dm, beta, mmin, delta_m):
    prob = p2_no_m1(mvec, beta, mmin, delta_m)
    cumulative_integral = xp.append(0,trapz_cumsum(prob, dx=dm, axis=-1))
    return lambda m: xp.interp(m, mvec, cumulative_integral)

def construct_K(mvec, dm, p1probs, beta, mmin, delta_m):
    Ifunc = construct_I(mvec, dm, beta, mmin, delta_m)
    prob = xp.nan_to_num(p1probs / Ifunc(mvec))
    cumulative_integral = xp.flip(xp.append(0,trapz_cumsum(prob, dx=dm, axis=-1, flipped=True)),axis=-1) # flip back
    return lambda m: xp.interp(m, mvec, cumulative_integral)

def ligo_combined_pm(m, mvec,dm, truths, return_pone_ptwo=False):
    pm1 =ligo_ppop(mvec, parameters=truths)  # cache + get p1 normalisation
    pm1_norm = trapz(pm1, dx=dm)
    pone = p1(m, pm1_norm, truths)

    Kfunc = construct_K(mvec, dm, pm1, truths["beta"], truths["mmin"], truths["delta_m"])
    ptwo = p2_total(m, Kfunc, pm1_norm, truths["beta"], truths["mmin"], truths["delta_m"])

    if return_pone_ptwo:
        return 0.5*(pone + ptwo), pone, ptwo
    else:
        return 0.5*(pone + ptwo)

def p2_no_m1_vectorized(m, beta, mmin, delta_m):
    return m**beta[:,None] * smoothing(m, mmin, delta_m)

def p2_total_vectorized(m, Kfunc, p1norm, beta, mmin, delta_m):
    return p2_no_m1_vectorized(m, beta, mmin, delta_m)/p1norm * Kfunc(m) # integral is a function of m

def construct_I_vectorized(mvec, dm, beta, mmin, delta_m):
    prob = p2_no_m1_vectorized(mvec, beta, mmin, delta_m)
    cumulative_integral = xp.concatenate((xp.zeros(prob.shape[0])[:,None],trapz_cumsum(prob, dx=dm, axis=-1)),axis=1)
    return lambda m: interp2d(m, [mvec.min(), mvec.max()], mvec.size, cumulative_integral)

def construct_K_vectorized(mvec, dm, p1probs, beta, mmin, delta_m):
    Ifunc = construct_I_vectorized(mvec, dm, beta, mmin, delta_m)
    prob = xp.nan_to_num(p1probs / Ifunc(mvec))
    cumulative_integral = xp.flip(xp.concatenate((xp.zeros(prob.shape[0])[:,None],trapz_cumsum(prob, dx=dm, axis=-1, flipped=True)), axis=1), axis=-1) # flip back
    return lambda m: interp2d(m, [mvec.min(), mvec.max()], mvec.size, cumulative_integral)

def ligo_combined_pm_vectorized(m, mvec,dm, truths, return_pone_ptwo=False):
    pm1 =ligo_ppop(mvec, parameters=truths)  # cache + get p1 normalisation
    pm1_norm = trapz(pm1, dx=dm, axis=-1)
    pone = p1(m, pm1_norm[:,None], truths)

    Kfunc = construct_K_vectorized(mvec, dm, pm1, truths["beta"], truths["mmin"], truths["delta_m"])
    ptwo = p2_total_vectorized(m, Kfunc, pm1_norm[:,None], truths["beta"], truths["mmin"], truths["delta_m"])

    if return_pone_ptwo:
        return 0.5*(pone + ptwo), pone, ptwo
    else:
        return 0.5*(pone + ptwo)
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
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

def truncnorm(xx, mu, sigma, high, low):
    # breakpoint()
    x_op = xp.repeat(xx, mu.size).reshape((xx.size,mu.size)).T
    hi_op = xp.repeat(high, xx.size).reshape((high.size,xx.size))
    lo_op = xp.repeat(low, xx.size).reshape((low.size,xx.size))

    norm = 2**0.5 / np.pi**0.5 / sigma
    norm /= erf((high - mu) / 2**0.5 / sigma) + erf((mu - low) / 2**0.5 / sigma)  #vector of norms
    try:
        prob = xp.exp(-xp.power(xx - mu[None,:], 2) / (2 * sigma[None,:]**2)) # array of dims len(xx) * len(mu)
        prob *= norm[None,:]  # should be fine considering dimensionality
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
        
        out =  xx**lam[None,:] * norm[None,:] # array of dims len(xx) * len(lam)
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
        ans = (1 + smooth_exp(masses - mmin[None,:], delta_m[None,:]))**-1
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
    mass, alpha, mmin, mmax, lam, mpp, sigpp, gaussian_mass_maximum=200
):
    p_pow = powerlaw(mass, -alpha, mmin, mmax)  # 2d array, dims N_samp * N_hp
    p_norm = truncnorm(mass, mu=mpp, sigma=sigpp, high=xp.asarray(gaussian_mass_maximum), low=mmin)  # same as above
    try:
        prob = (1 - lam[None,:]) * p_pow + lam[None,:] * p_norm
        
    except IndexError:
        prob = (1 - lam) * p_pow + lam * p_norm
    return prob

def p1(m, mgrid, truths):
    p = ligo_ppop(m, truths)
    p1norm = trapz(ligo_ppop(mgrid, parameters=truths), dx=mgrid[1]-mgrid[0])
    return p/p1norm

def p1_grid(mgrid, truths):
    p = ligo_ppop(mgrid, truths)
    p1norm = trapz(p, x=mgrid)
    return (p.T/p1norm).T

def qfunc(q, m1, beta, mmin, delta_m):
    return q**beta * smoothing(q*m1, mmin, delta_m) # not normalised

def pq(q, m1, truths):
    norm_interp = construct_qnorm_interpolator(truths)
    probs = qfunc(q[:,None], m1[None,:], truths["beta"][None,:], truths["mmin"][None,:],truths["delta_m"][None,:])
    norm = norm_interp(m1)
    return xp.nan_to_num(xp.squeeze(probs/norm[None,:]))

def construct_qnorm_interpolator(truths):
    qvec = xp.linspace(truths["mmin"]/100, 1, 1000)
    mvec = xp.linspace(truths["mmin"],100,1000)
    probs = qfunc(qvec[:,None], mvec[None,:], truths["beta"][None,:], truths["mmin"][None,:],truths["delta_m"][None,:])
    integrated = trapz(probs, dx=qvec[1]-qvec[0], axis=0)
    return lambda x: xp.interp(x, mvec, integrated)

def sample_snakes_cdf(size, cdfs, sample_vec):
    arange = xp.arange(np.shape(cdfs)[1])
    cdf_snake = (cdfs + arange[None,:]).T.flatten()# (q_grid_size * size)
    samples_all = (sample_vec[:,None] + arange[None,:]).T.flatten()
    samples = xp.interp(xp.random.random(size), cdf_snake, samples_all) - arange# (size)
    return samples

def sample_m1_q_fast(truths, size, mmax=200, m1_grid_size=1000, q_grid_size = 250):
    mvec = xp.linspace(truths['mmin'],mmax, m1_grid_size)# (m1_grid_size,)
    probs = p1_grid(mvec, truths)
    print(probs)
    cdf = xp.cumsum(probs,axis = 1)# (m1_grid_size, )
    cdf /= cdf.max()
    
    m1samples = sample_snakes_cdf(size, cdf, mvec)
    
    
    # m1samples = xp.interp(xp.random.random(size), cdf, mvec)# (size, )

    qvec = xp.linspace(truths['mmin']/mmax, 1, q_grid_size)# (q_grid_size, )
    q_curves = pq(qvec, m1samples, truths) # (q_grid_size, size)
    cdfs = xp.nan_to_num(xp.cumsum(q_curves, axis=1) / xp.sum(q_curves,axis=1))# (q_grid_size, size)
    arange = xp.arange(size)
    cdf_snake = (cdfs + arange[None,:]).T.flatten()# (q_grid_size * size)
    sample_snake = xp.random.random(size) + arange
    q_all = (qvec[:,None] + arange[None,:]).T.flatten()# (q_grid_size * size)
    qsamples = xp.interp(sample_snake, cdf_snake, q_all) - arange# (size)

    return m1samples, qsamples*m1samples

def sample_m1_q(truths, size, mmax=200, grid_size=1000):
    qsamples = xp.zeros(size)
    mvec = xp.linspace(truths['mmin'],mmax, grid_size)
    probs = p1_grid(mvec, truths)
    cdf = xp.cumsum(probs)
    cdf /= cdf.max()
    m1samples = xp.interp(xp.random.random(size), cdf, mvec)
    qvec = xp.linspace(truths['mmin']/mmax, 1, grid_size)
    q_curves = pq(qvec, m1samples, truths)
    cdfs = xp.nan_to_num(xp.cumsum(q_curves, axis=0) / xp.sum(q_curves,axis=0))
    arange = xp.arange(size)
    cdf_snake = (cdfs + arange[None,:]).T.flatten()
    sample_snake = xp.random.random(size) + arange
    q_all = (qvec[:,None] + arange[None,:]).T.flatten()
    qsamples = xp.interp(sample_snake, cdf_snake, q_all) - arange

    return m1samples, qsamples
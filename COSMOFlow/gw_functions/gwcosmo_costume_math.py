'''
This module collects analytical and numerical probability density functions.
'''

import numpy as _np
from scipy.stats import truncnorm as _truncnorm
import copy as _copy
from scipy.interpolate import interp1d as _interp1d
from scipy.special import erf as _erf
from scipy.special import logsumexp as _logsumexp
# from scipy.integrate import cumtrapz as _cumtrapz
from scipy.integrate import cumulative_trapezoid as _cumtrapz

def _S_factor(mass, mmin,delta_m):
    '''
    This function return the value of the window function defined as Eqs B6 and B7 of https://arxiv.org/pdf/2010.14533.pdf

    Parameters
    ----------
    mass: np.array or float
        array of x or masses values
    mmin: float or np.array (in this case len(mmin) == len(mass))
        minimum value of window function
    delta_m: float or np.array (in this case len(delta_m) == len(mass))
        width of the window function

    Returns
    -------
    Values of the window function
    '''

    if not isinstance(mass,_np.ndarray):
        mass = _np.array([mass])

    to_ret = _np.ones_like(mass)
    if delta_m == 0:
        return to_ret

    mprime = mass-mmin

    # Defines the different regions of thw window function ad in Eq. B6 of  https://arxiv.org/pdf/2010.14533.pdf
    select_window = (mass>mmin) & (mass<(delta_m+mmin))
    select_one = mass>=(delta_m+mmin)
    select_zero = mass<=mmin

    effe_prime = _np.ones_like(mass)

    # Definethe f function as in Eq. B7 of https://arxiv.org/pdf/2010.14533.pdf
    effe_prime[select_window] = _np.exp(_np.nan_to_num((delta_m/mprime[select_window])+(delta_m/(mprime[select_window]-delta_m))))
    to_ret = 1./(effe_prime+1)
    to_ret[select_zero]=0.
    to_ret[select_one]=1.
    return to_ret

def get_PL_norm(alpha,minv,maxv):
    '''
    This function returns the powerlaw normalization factor

    Parameters
    ----------
    alpha: float
        Powerlaw slope
    min_pl: float
        lower cutoff
    max_pl: float
        upper cutoff
    '''

    # Get the PL norm as in Eq. 24 on the tex document
    if alpha == -1:
        return _np.log(maxv/minv)
    else:
        return (_np.power(maxv,alpha+1) - _np.power(minv,alpha+1))/(alpha+1)

def get_gaussian_norm(mu,sigma,min,max):
    '''
    This function returns the gaussian normalization factor

    Parameters
    ----------
    mu: float
        mean of the gaussian
    sigma: float
        standard deviation of the gaussian
    min_pl: float
        lower cutoff
    max_pl: float
        upper cutoff
    '''

    # Get the gaussian norm as in Eq. 28 on the tex document
    max_point = (max-mu)/(sigma*_np.sqrt(2.))
    min_point = (min-mu)/(sigma*_np.sqrt(2.))
    return 0.5*_erf(max_point)-0.5*_erf(min_point)


class SmoothedProb(object):
    '''
    Class for smoothing the low part of a PDF. The smoothing follows Eq. B7 of
    2010.14533.

    Parameters
    ----------
    origin_prob: class
        Original prior class to smooth from this module
    bottom: float
        minimum cut-off. Below this,gw_functions.gwcosmo_priors the window is 0.
    bottom_smooth: float
        smooth factor. The smoothing acts between bottom and bottom+bottom_smooth
    '''

    def __init__(self,origin_prob,bottom,bottom_smooth):

        self.origin_prob = _copy.deepcopy(origin_prob)
        self.bottom_smooth = bottom_smooth
        self.bottom = bottom
        self.maximum=self.origin_prob.maximum
        self.minimum=self.origin_prob.minimum

        # Find the values of the integrals in the region of the window function before and after the smoothing
        int_array = _np.linspace(bottom,bottom+bottom_smooth,1000)
        integral_before = _np.trapz(self.origin_prob.prob(int_array),int_array)
        integral_now = _np.trapz(self.prob(int_array),int_array)

        self.integral_before = integral_before
        self.integral_now = integral_now
        # Renormalize the the smoother function.
        self.norm = 1 - integral_before + integral_now - self.origin_prob.cdf(bottom)

        x_eval = _np.logspace(_np.log10(bottom),_np.log10(bottom+bottom_smooth),1000)
        cdf_numeric = _cumtrapz(self.prob(x_eval),x_eval)
        self.cached_cdf_window = _interp1d(x_eval[:-1:],cdf_numeric,fill_value='extrapolate',bounds_error=False,kind='cubic')

        x_eval = _np.logspace(_np.log10(bottom),_np.log10(bottom+bottom_smooth),1000)
        cdf_numeric = _cumtrapz(self.prob(x_eval),x_eval)
        self.cached_cdf_window = _interp1d(x_eval[:-1:],cdf_numeric,fill_value='extrapolate',bounds_error=False,kind='cubic')

    def prob(self,x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        return _np.exp(self.log_prob(x))

    def log_prob(self,x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        # Return the window function
        window = _S_factor(x, self.bottom,self.bottom_smooth)

        if hasattr(self,'norm'):
            prob_ret =self.origin_prob.log_prob(x)+_np.log(window)-_np.log(self.norm)
        else:
            prob_ret =self.origin_prob.log_prob(x)+_np.log(window)

        return prob_ret

    def log_conditioned_prob(self,x,a,b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array
            Value at which compute the probability
        a: np.array
            New lower boundary
        b: np.array
            New upper boundary
        """

        to_ret = self.log_prob(x)
        # Find the new normalization in the new interval
        new_norm = self.cdf(b)-self.cdf(a)
        # Apply the new normalization and put to zero all the values above/below the interval
        to_ret-=_np.log(new_norm)
        to_ret[(x<a) | (x>b)] = -_np.inf

        return to_ret

    def cdf(self,x):
        """
        Returns the cumulative density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the cumulative
        """

        to_ret = _np.ones_like(x)
        to_ret[x<self.bottom] = 0.
        to_ret[(x>=self.bottom) & (x<=(self.bottom+self.bottom_smooth))] = self.cached_cdf_window(x[(x>=self.bottom) & (x<=(self.bottom+self.bottom_smooth))])
        to_ret[x>=(self.bottom+self.bottom_smooth)]=(self.integral_now+self.origin_prob.cdf(
        x[x>=(self.bottom+self.bottom_smooth)])-self.origin_prob.cdf(self.bottom+self.bottom_smooth))/self.norm

        return to_ret

    def conditioned_prob(self,x,a,b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array
            Value at which compute the probability
        a: np.array
            New lower boundary
        b: np.array
            New upper boundary
        """

        return _np.exp(self.log_conditioned_prob(x,a,b))

class PowerLaw_math(object):
    """
    Class for a powerlaw probability :math:`p(x) \\propto x^{\\alpha}` defined in
    [a,b]

    Parameters
    ----------
    alpha: float
        Powerlaw slope
    min_pl: float
        lower cutoff
    max_pl: float
        upper cutoff
    """

    def __init__(self,alpha,min_pl,max_pl):

        self.minimum = min_pl
        self.maximum = max_pl
        self.min_pl = min_pl
        self.max_pl = max_pl
        self.alpha = alpha

        # Get the PL norm and as Eq. 24 on the paper
        self.norm = get_PL_norm(alpha,min_pl,max_pl)

    def prob(self,x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        return _np.exp(self.log_prob(x))

    def log_prob(self,x):
        """
        Returns the logarithm of the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        to_ret = self.alpha*_np.log(x)-_np.log(self.norm)
        to_ret[(x<self.min_pl) | (x>self.max_pl)] = -_np.inf

        return to_ret

    def log_conditioned_prob(self,x,a,b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New upper boundary
        """

        norms = get_PL_norm(self.alpha,a,b)
        to_ret = self.alpha*_np.log(x)-_np.log(norms)
        to_ret[(x<a) | (x>b)] = -_np.inf

        return to_ret

    def cdf(self,x):
        """
        Returns the cumulative density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the cumulative
        """

        # Define the cumulative density function, see  Eq. 24 to see the integral form

        if self.alpha == -1:
            to_ret = _np.log(x/self.min_pl)/self.norm
        else:
            to_ret =((_np.power(x,self.alpha+1)-_np.power(self.min_pl,self.alpha+1))/(self.alpha+1))/self.norm

        to_ret *= (x>=self.min_pl)

        if hasattr(x, "__len__"):
            to_ret[x>self.max_pl]=1.
        else:
            if x>self.max_pl : to_ret=1.

        return to_ret

    def conditioned_prob(self,x,a,b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New upper boundary
        """

        return _np.exp(self.log_conditioned_prob(x,a,b))



class Truncated_Gaussian_math(object):
    """
    Class for a truncated gaussian in
    [a,b]

    Parameters
    ----------
    mu: float
        mean of the gaussian
    sigma: float
        standard deviation of the gaussian
    min_g: float
        lower cutoff
    max_g: float
        upper cutoff
    """

    def __init__(self,mu,sigma,min_g,max_g):

        self.minimum = min_g
        self.maximum = max_g
        self.max_g=max_g
        self.min_g=min_g
        self.mu = mu
        self.sigma=sigma

        # Find the gaussian normalization as in Eq. 28 in the tex document
        self.norm = get_gaussian_norm(mu,sigma,min_g,max_g)

    def prob(self,x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        return _np.exp(self.log_prob(x))


    def log_prob(self,x):
        """
        Returns the logarithm of the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        to_ret = -_np.log(self.sigma)-0.5*_np.log(2*_np.pi)-0.5*_np.power((x-self.mu)/self.sigma,2.)-_np.log(self.norm)
        to_ret[(x<self.min_g) | (x>self.max_g)] = -_np.inf

        return to_ret

    def log_conditioned_prob(self,x,a,b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New upper boundary
        """

        norms = get_gaussian_norm(self.mu,self.sigma,a,b)
        to_ret = -_np.log(self.sigma)-0.5*_np.log(2*_np.pi)-0.5*_np.power((x-self.mu)/self.sigma,2.)-_np.log(norms)
        to_ret[(x<a) | (x>b)] = -_np.inf

        return to_ret


    def cdf(self,x):
        """
        Returns the cumulative density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the cumulative
        """

        # Define the cumulative density function as in Eq. 28 on the paper to see the integral form

        max_point = (x-self.mu)/(self.sigma*_np.sqrt(2.))
        min_point = (self.min_g-self.mu)/(self.sigma*_np.sqrt(2.))

        to_ret = (0.5*_erf(max_point)-0.5*_erf(min_point))/self.norm

        to_ret *= (x>=self.min_g)

        if hasattr(x, "__len__"):
            to_ret[x>self.max_g]=1.
        else:
            if x>self.max_g : to_ret=1.

        return to_ret

    def conditioned_prob(self,x,a,b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New upper boundary
        """

        return _np.exp(self.log_conditioned_prob(x,a,b))



class PowerLawGaussian_math(object):
    """
    Class for a powerlaw probability plus gausian peak
    :math:`p(x) \\propto (1-\\lambda)x^{\\alpha}+\\lambda \\mathcal{N}(\\mu,\\sigma)`. Each component is defined in
    a different interval

    Parameters
    ----------
    alpha: float
        Powerlaw slope
    min_pl: float
        lower cutoff
    max_pl: float
        upper cutoff
    lambda_g: float
        fraction of prob coming from gaussian peak
    mean_g: float
        mean for the gaussian
    sigma_g: float
        standard deviation for the gaussian
    min_g: float
        minimum for the gaussian component
    max_g: float
        maximim for the gaussian component
    """

    def __init__(self,alpha,min_pl,max_pl,lambda_g,mean_g,sigma_g,min_g,max_g):

        self.minimum = _np.min([min_pl,min_g])
        self.maximum = _np.max([max_pl,max_g])

        self.lambda_g=lambda_g

        self.pl= PowerLaw_math(alpha,min_pl,max_pl)
        self.gg = Truncated_Gaussian_math(mean_g,sigma_g,min_g,max_g)


    def prob(self,x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        # Define the PDF as in Eq. 36-37-38 on on the tex document
        return _np.exp(self.log_prob(x))

    def cdf(self,x):
        """
        Returns the cumulative density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the cumulative
        """

        return (1-self.lambda_g)*self.pl.cdf(x)+self.lambda_g*self.gg.cdf(x)

    def conditioned_prob(self,x,a,b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New upper boundary
        """

        return _np.exp(self.log_conditioned_prob(x,a,b))

    def log_prob(self,x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        # Define the PDF as in Eq. 36-37-38 on on the tex document
        return _np.logaddexp(_np.log1p(-self.lambda_g)+self.pl.log_prob(x),_np.log(self.lambda_g)+self.gg.log_prob(x))

    def log_conditioned_prob(self,x,a,b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New upper boundary
        """

        return  _np.logaddexp(_np.log1p(-self.lambda_g)+self.pl.log_conditioned_prob(x,a,b),_np.log(self.lambda_g)+self.gg.log_conditioned_prob(x,a,b))


class PowerLawDoubleGaussian_math(object):
    """
    Class for a powerlaw probability plus gausian peak
    :math:`p(x) \\propto (1-\\lambda)x^{\\alpha}+\\lambda \\lambda_1 \\mathcal{N}(\\mu_1,\\sigma_1)+\\lambda (1-\\lambda_1) \\mathcal{N}(\\mu_2,\\sigma_2)`.
    Each component is defined ina different interval

    Parameters
    ----------
    alpha: float
        Powerlaw slope
    min_pl: float
        lower cutoff
    max_pl: float
        upper cutoff
    lambda_g: float
        fraction of prob coming in any gaussian peak
    lambda_g_low: float
        fraction of prob in lower gaussian peak
    mean_g_low: float
        mean for the gaussian
    sigma_g_low: float
        standard deviation for the gaussian# Define the PDF as in Eq. 37 on on the tex document
    mean_g_high: float
        mean for the gaussian
    sigma_g_high: float
        standard deviation for the gaussian
    min_g: float
        minimum for the gaussian components
    max_g: float
        maximim for the gaussian components
    """

    def __init__(self,alpha,min_pl,max_pl,lambda_g, lambda_g_low,
    mean_g_low,sigma_g_low,mean_g_high,sigma_g_high,min_g,max_g):

        self.minimum = _np.min([min_pl,min_g])
        self.maximum = _np.max([max_pl,max_g])

        self.lambda_g = lambda_g
        self.lambda_g_low = lambda_g_low

        self.pl= PowerLaw_math(alpha,min_pl,max_pl)
        self.gg_low = Truncated_Gaussian_math(mean_g_low,sigma_g_low,min_g,max_g)
        self.gg_high = Truncated_Gaussian_math(mean_g_high,sigma_g_high,min_g,max_g)

    def prob(self,x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        return _np.exp(self.log_prob(x))


    def log_prob(self,x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        # Define the PDF as in Eq. 44-45-46 on the tex document

        pl_part = _np.log1p(-self.lambda_g)+self.pl.log_prob(x)
        g_low = self.gg_low.log_prob(x)+_np.log(self.lambda_g)+_np.log(self.lambda_g_low)
        g_high = self.gg_high.log_prob(x)+_np.log(self.lambda_g)+_np.log1p(-self.lambda_g_low)

        return _logsumexp(_np.stack([pl_part,g_low,g_high]),axis=0)

    def log_conditioned_prob(self,x,a,b):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        # Define the PDF as in Eq.  44-45-46  on the tex document

        pl_part = _np.log1p(-self.lambda_g)+self.pl.log_conditioned_prob(x,a,b)
        g_low = self.gg_low.log_conditioned_prob(x,a,b)+_np.log(self.lambda_g)+_np.log(self.lambda_g_low)
        g_high = self.gg_high.log_conditioned_prob(x,a,b)+_np.log(self.lambda_g)+_np.log1p(-self.lambda_g_low)

        return _logsumexp(_np.stack([pl_part,g_low,g_high]),axis=0)

    def cdf(self,x):
        """
        Returns the cumulative density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the cumulative
        """

        pl_part = (1-self.lambda_g)*self.pl.cdf(x)
        g_part =self.gg_low.cdf(x)*self.lambda_g*self.lambda_g_low+self.gg_high.cdf(x)*self.lambda_g*(1-self.lambda_g_low)

        return pl_part+g_part

    def conditioned_prob(self,x,a,b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New upper boundary
        """

        return _np.exp(self.log_conditioned_prob(x,a,b))



class BrokenPowerLaw_math(object):
    """
    Class for a broken powerlaw probability
    :math:`p(x) \\propto x^{\\alpha}` if :math:`min<x<b(max-min)`, :math:`p(x) \\propto x^{\\beta}` if :math:`b(max-min)<x<max`.

    Parameters
    ----------
    alpha_1: float
        Powerlaw slope for first component
    alpha_2: float
        Powerlaw slope for second component
    min_pl: float
        lower cutoff
    max_pl: float
        upper cutoff
    b: float
        fraction in [0,1] at which the powerlaw breaks
    """

    def __init__(self,alpha_1,alpha_2,min_pl,max_pl,b):

        self.minimum = min_pl
        self.maximum = max_pl

        self.min_pl = min_pl
        self.max_pl = max_pl

        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2

        # Define the breaking point
        self.break_point = min_pl+b*(max_pl-min_pl)
        self.b=b

        # Initialize the single powerlaws
        self.pl1=PowerLaw_math(alpha_1,min_pl,self.break_point)
        self.pl2=PowerLaw_math(alpha_2,self.break_point,max_pl)

        # Define the broken powerlaw as in Eq. 39-40-41 on the tex document
        self.new_norm=(1+self.pl1.prob(_np.array([self.break_point]))/self.pl2.prob(_np.array([self.break_point])))

    def prob(self,x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        # Define the PDF as in Eq. 39-40-41 on the tex document
        return _np.exp(self.log_prob(x))


    def log_prob(self,x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        # Define the PDF as in Eq. 39-40-41 on the tex document

        to_ret = _np.logaddexp(self.pl1.log_prob(x),self.pl2.log_prob(x)+self.pl1.log_prob(_np.array([self.break_point]))
        -self.pl2.log_prob(_np.array([self.break_point])))-_np.log(self.new_norm)
        return to_ret

    def log_conditioned_prob(self,x,a,b):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        # Define the PDF as in Eq. 39-40-41 on the tex document

        to_ret = _np.logaddexp(self.pl1.log_conditioned_prob(x,a,b),self.pl2.log_conditioned_prob(x,a,b)
        +self.pl1.log_prob(_np.array([self.break_point]))-self.pl2.log_prob(_np.array([self.break_point])))-_np.log(self.new_norm)

        return to_ret

    def cdf(self,x):
        """
        Returns the cumulative density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the cumulative
        """

        return (self.pl1.cdf(x)+self.pl2.cdf(x)*(self.pl1.prob(_np.array([self.break_point]))/self.pl2.prob(_np.array([self.break_point]))))/self.new_norm

    def conditioned_prob(self,x,a,b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New upper boundary
        """

        return _np.exp(self.log_conditioned_prob(x,a,b))

import cupy as xp
from cupyx.scipy.special import erf
import numpy as np
from scipy.stats import truncnorm

class MassPrior:
    """
    A class to represent the mass prior distribution for binary black holes.

    Attributes:
        parameters (dict): Population parameters.
        mgrid (int): Number of grid points.
        mmax_total (float): Maximum total mass.
        population_type (str): Type of population.
        m_vect (cupy.ndarray): Vector of mass values.
        log_m_vect (cupy.ndarray): Vector of log mass values.
        dm (float): Mass grid spacing.
    """
    def __init__(self, population_parameters, mmax_total=200, mgrid=200):
        self.parameters = population_parameters
        self.mgrid = mgrid
        self.mmax_total = mmax_total
        self.population_type = population_parameters['name']
        self.m_vect = xp.linspace(0, self.mmax_total, self.mgrid)
        self.log_m_vect = xp.exp(xp.linspace(xp.log(1), xp.log(self.mmax_total), self.mgrid))
        self.dm = xp.diff(self.m_vect)[0]

    def gaussian_peak(self, xx, mu, sigma, high, low):
        """
        Compute the Gaussian peak probability.

        Args:
            xx (cupy.ndarray): Mass values.
            mu (float): Mean of the Gaussian.
            sigma (float): Standard deviation of the Gaussian.
            high (float): Upper limit for the Gaussian.
            low (float): Lower limit for the Gaussian.

        Returns:
            cupy.ndarray: Probability values.
        """
        norm = 2**0.5 / np.pi**0.5 / sigma
        norm /= erf((high - mu) / 2**0.5 / sigma) + erf((mu - low) / 2**0.5 / sigma)
        prob = xp.exp(-xp.power(xx - mu, 2) / (2 * sigma**2))
        prob *= norm
        
        if xp.size(xx) > 1: 
            prob[xx > high] = 0.0
            prob[xx < low] = 0.0
        elif (xx > high) and (xx < low):
            prob = 0 
        
        return prob
    
    def power_law(self, xx, index, high, low):
        """
        Compute the power-law probability.

        Args:
            xx (cupy.ndarray): Mass values.
            index (float): Power-law index.
            high (float): Upper limit for the power-law.
            low (float): Lower limit for the power-law.

        Returns:
            cupy.ndarray: Probability values.
        """
        norm = (high**(-index+1) - low**(-index+1))/(-index + 1)
        power = xx**(-index)/norm 
        if xp.size(xx) > 1: 
            power[xx > high] = 0.0
            power[xx < low] = 0.0
        elif (xx > high) and (xx < low):
            power = 0 
        return power
    
    def smooth_factor(self, xx, mmin, smooth_scale):
        """
        Compute the smoothing factor.

        Args:
            xx (cupy.ndarray): Mass values.
            mmin (float): Minimum mass.
            smooth_scale (float): Smoothing scale.

        Returns:
            cupy.ndarray: Smoothing factor values.
        """
        mprime = xx - mmin
        smooth_exp = xp.exp((smooth_scale/mprime)+(smooth_scale/(mprime-smooth_scale)))
        ans = (1 + smooth_exp)**-1

        all_zero_rows = np.all(ans == 0, axis=1)
        # Change those rows to all ones
        ans[all_zero_rows] = 1
        
        if xp.size(xx) > 1: 
            ans[xx > mmin + smooth_scale] = 1.0
            ans[xx < mmin] = 0.0

        else:
            if xx > mmin + smooth_scale:
                ans = 1
            elif xx < mmin:
                ans = 0 
        return ans
    
    def powerlaw_plus_peak_smooth(self, xx):
        """
        Compute the combined power-law and Gaussian peak with smoothing.

        Args:
            xx (cupy.ndarray): Mass values.

        Returns:
            cupy.ndarray: Probability values.
        """
        factor1 = self.power_law(xx, self.parameters['alpha'], self.parameters['mmax'], self.parameters['mmin'])
        factor2 = self.gaussian_peak(xx, self.parameters['mu_g'], self.parameters['sigma_g'], self.parameters['mmax'], self.parameters['mmin'])
        return (factor1*(1-self.parameters['lambda_peak']) + factor2*(self.parameters['lambda_peak'])) * self.smooth_factor(xx, self.parameters['mmin'], self.parameters['delta_m'])
    
    def powerlaw_plus_peak_smooth_2gaussian(self, xx):
        """
        Compute the combined power-law and two Gaussian peaks with smoothing.

        Args:
            xx (cupy.ndarray): Mass values.

        Returns:
            cupy.ndarray: Probability values.
        """
        factor1 = self.power_law(xx, self.parameters['alpha'], self.parameters['mmax'], self.parameters['mmin'])
        factor2 = self.gaussian_peak(xx, self.parameters['mu_g'], self.parameters['sigma_g'], self.parameters['mmax'], self.parameters['mmin'])
        factor3 = self.gaussian_peak(xx, self.parameters['mu_g2'], self.parameters['sigma_g2'], self.parameters['mmax'], self.parameters['mmin'])
        return (factor1 * (1 - self.parameters['lambda_peak']) + (self.parameters['lambda_peak']) * (factor2 * (1 - self.parameters['lambda_peak2']) + factor3 * self.parameters['lambda_peak2'])) * self.smooth_factor(xx, self.parameters['mmin'], self.parameters['delta_m'])
    
    def powerlaw_plus_peak_smooth_vect(self, xx):
        """
        Compute the combined power-law and Gaussian peak with smoothing for vectors.

        Args:
            xx (cupy.ndarray): Mass values.

        Returns:
            cupy.ndarray: Probability values.
        """
        factor1 = self.power_law(xx[None, :], self.parameters['alpha'][:, None], self.parameters['mmax'][:, None], self.parameters['mmin'][:, None])
        factor2 = self.gaussian_peak(xx[None, :], self.parameters['mu_g'][:, None], self.parameters['sigma_g'][:, None], self.parameters['mmax'][:, None], self.parameters['mmin'][:, None])
        return (factor1 * (1 - self.parameters['lambda_peak'][:, None]) + factor2 * (self.parameters['lambda_peak'][:, None])) * self.smooth_factor(xx[None, :], self.parameters['mmin'][:, None], self.parameters['delta_m'][:, None])

    
    
    def powerlaw_smooth_m2_vect(self, xx, m1, beta=None, mmin=None, delta_m=None):
        """
        Compute the power-law probability with smoothing for m2 vectors.

        Args:
            xx (cupy.ndarray): Mass values.
            m1 (cupy.ndarray): Primary mass values.
            beta (cupy.ndarray, optional): Power-law index.
            mmin (cupy.ndarray, optional): Minimum mass.
            delta_m (cupy.ndarray, optional): Smoothing scale.

        Returns:
            cupy.ndarray: Probability values.
        """
        if beta is None:
            beta = self.parameters['beta'][:, None]
            mmin = self.parameters['mmin'][:, None]
            delta_m = self.parameters['delta_m'][:, None]
        else:
            beta = xp.array(beta[:, None])
            mmin = xp.array(mmin[:, None])
            delta_m = xp.array(delta_m[:, None])
        
        factor1 = self.power_law(xx, beta, m1[:, None], mmin)
        return factor1 * self.smooth_factor(xx, mmin, delta_m)

    def powerlaw_plus_peak_smooth_2gaussian_vect(self, xx):
        """
        Compute the combined power-law and two Gaussian peaks with smoothing for vectors.
    
        Args:
            xx (cupy.ndarray): Mass values.
    
        Returns:
            cupy.ndarray: Probability values.
        """
        # Calculating the power law factor for the given parameters in a vectorized form
        factor1 = self.power_law(
            xx[None, :], 
            self.parameters['alpha'][:, None], 
            self.parameters['mmax'][:, None], 
            self.parameters['mmin'][:, None]
        )
        
        # Calculating the first Gaussian peak factor for the given parameters in a vectorized form
        factor2 = self.gaussian_peak(
            xx[None, :], 
            self.parameters['mu_g'][:, None], 
            self.parameters['sigma_g'][:, None], 
            self.parameters['mmax'][:, None], 
            self.parameters['mmin'][:, None]
        )
        
        # Calculating the second Gaussian peak factor for the given parameters in a vectorized form
        factor3 = self.gaussian_peak(
            xx[None, :], 
            self.parameters['mu_g2'][:, None], 
            self.parameters['sigma_g2'][:, None], 
            self.parameters['mmax'][:, None], 
            self.parameters['mmin'][:, None]
        )
        
        # Combining the power law and both Gaussian peaks, applying the lambda factors for weighting
        combined = (
            factor1 * (1 - self.parameters['lambda_peak'][:, None]) +
            (self.parameters['lambda_peak'][:, None]) * (
                factor2 * (1 - self.parameters['lambda_peak2'][:, None]) + 
                factor3 * self.parameters['lambda_peak2'][:, None]
            )
        )
        
        # Applying the smoothing factor to the combined distribution
        result = combined * self.smooth_factor(
            xx[None, :], 
            self.parameters['mmin'][:, None], 
            self.parameters['delta_m'][:, None]
        )
        
        return result

    
    
    def make_cdfs_m2(self, m1, m_array_long=None, prior_pdf=None):
        """
        Compute the CDFs for m2.

        Args:
            m1 (cupy.ndarray): Primary mass values.
            m_array_long (cupy.ndarray, optional): Long array of mass values.
            prior_pdf (cupy.ndarray, optional): Prior probability density function.

        Returns:
            cupy.ndarray: CDF values.
        """
        m_vect_m2 = xp.linspace(0, m1[:, None], self.mgrid, axis=1)
        m_vect_m2 = xp.reshape(m_vect_m2, (len(m1), self.mgrid))
        # print(xp.shape(m_vect_m2))
        if prior_pdf is None:
            if m_array_long is None:
                pdf = self.powerlaw_smooth_m2_vect(m_vect_m2, m1)
            else:
                pdf = self.powerlaw_smooth_m2_vect(m_array_long, m1)
        else:
            pdf = prior_pdf
        cdf = xp.cumsum(pdf, axis=1)
        cdf_maximum = xp.amax(cdf, axis=1)[:, None]
        return cdf / cdf_maximum
    
    def make_cdfs(self, log_space=False, m1_arr=None, pdf_function = None):
        """
        Compute the CDFs.

        Args:
            log_space (bool, optional): Whether to use log space.
            m1_arr (cupy.ndarray, optional): Array of primary mass values.

        Returns:
            cupy.ndarray: CDF values.
        """
        if pdf_function is None:
            pdf_function = self.powerlaw_plus_peak_smooth_vect()
        
        if m1_arr is None:
            if log_space:
                arr_m = self.log_m_vect
            else:
                arr_m = self.m_vect
        pdf = pdf_function(arr_m)
        cdf = xp.cumsum(pdf, axis=1)
        cdf_maximum = xp.amax(cdf, axis=1)[:, None]
        return cdf / cdf_maximum
    
    def draw_m(self, Nsamples, cdfs, m_array_long=None, log_space=False, m2_arr = None):
        """
        Draw samples from the mass distribution.

        Args:
            Nsamples (int): Number of samples.
            cdfs (cupy.ndarray): CDF values.
            m_array_long (cupy.ndarray, optional): Long array of mass values.
            log_space (bool, optional): Whether to use log space.

        Returns:
            numpy.ndarray: Sampled mass values.
        """
        N = len(cdfs)
        cdfs_snake = xp.asarray(np.concatenate(cdfs)) 
        if m2_arr is None:
            if m_array_long is not None: 
                mlist = m_array_long
            else: 
                if log_space:
                    mlist = xp.ndarray.tolist(self.log_m_vect)
                else: 
                    mlist = xp.ndarray.tolist(self.m_vect)
                mlist = N * mlist
        else:
            mlist = m2_arr
        m_array = xp.asarray(mlist)
        cdfs_snake = cdfs_snake + xp.repeat(xp.arange(N), self.mgrid)
        t = xp.random.uniform(0, 1, size=N*Nsamples) + xp.repeat(xp.arange(N), Nsamples)
        return xp.interp(t, cdfs_snake, m_array).get()
    
    def draw_m_simple(self, Nsamples, cdf, m_array_long=None):
        """
        Draw samples from a simple mass distribution.

        Args:
            Nsamples (int): Number of samples.
            cdf (cupy.ndarray): CDF values.
            m_array_long (cupy.ndarray, optional): Long array of mass values.

        Returns:
            numpy.ndarray: Sampled mass values.
        """
        t = xp.random.uniform(0, 1, size=Nsamples)
        return xp.interp(t, cdf, self.m_vect).get()
    
    def m1_m2_sampling(self, N, Nselect, beta, mmin, delta_m, cdfs_m1, mass_class):
        """
        Perform sampling for m1 and m2.

        Args:
            N (int): Number of samples.
            Nselect (int): Number of selected samples.
            beta (cupy.ndarray): Power-law index.
            mmin (cupy.ndarray): Minimum mass.
            delta_m (cupy.ndarray): Smoothing scale.
            cdfs_m1 (cupy.ndarray): CDF values for m1.
            mass_class (MassPrior): Instance of MassPrior class.

        Returns:
            tuple: Sampled m1 and m2 values.
        """
        samples = mass_class.draw_m(Nselect, cdfs_m1)

        while True:
            inx_nan = (np.where((samples < mmin.get())))[0]
            if len(inx_nan) != 0:
                samples[inx_nan] = mass_class.draw_m(1, np.repeat(cdfs_m1, Nselect, axis=0)[inx_nan, :])
            else:
                break

        m1 = samples
        m2_arr = xp.array((xp.linspace(mmin, xp.array(samples), mass_class.mgrid))).T
        m2_arr_long = (m2_arr.reshape(int(N * Nselect * mass_class.mgrid), -1)).flatten()

        pdfs_m2 = mass_class.powerlaw_smooth_m2_vect(m2_arr, xp.array(samples), beta=beta, mmin=mmin, delta_m=delta_m)
        cdfs_m2 = mass_class.make_cdfs_m2(xp.array(samples), prior_pdf=pdfs_m2)
        samples_m2 = mass_class.draw_m(1, cdfs_m2, m_array_long=m2_arr_long)

        while True:
            inx_nan = np.where((np.isnan(samples_m2)) | (samples_m2 > samples))[0]
            if len(inx_nan) != 0:
                m2_arr = xp.linspace((mmin)[inx_nan], xp.array(samples[inx_nan]), mass_class.mgrid).T
                m2_arr_long = (m2_arr.reshape(int(len(inx_nan) * mass_class.mgrid), -1)).flatten()
                pdfs_m2[inx_nan] = mass_class.powerlaw_smooth_m2_vect(m2_arr, xp.array(samples[inx_nan]), beta=beta[inx_nan], mmin=mmin[inx_nan], delta_m=delta_m[inx_nan])
                cdfs_m2[inx_nan] = mass_class.make_cdfs_m2(xp.array(samples)[inx_nan], prior_pdf=pdfs_m2[inx_nan])
                samples_m2[inx_nan] = mass_class.draw_m(1, cdfs_m2[inx_nan], m_array_long=m2_arr_long)
            else:
                break

        m2 = samples_m2
        return m1, m2

    
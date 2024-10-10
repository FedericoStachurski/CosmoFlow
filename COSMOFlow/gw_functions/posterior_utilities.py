# Global Imports ####################### GWCOSMO CODE! #########################
import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz

from scipy.optimize import fmin
from scipy.interpolate import interp1d, UnivariateSpline


class confidence_interval(object):
    def __init__(self, posterior, param, level=0.683, verbose=False):
        self.posterior = posterior
        self.param = param
        self.level = level
        self.verbose = verbose
        self.lower_level, self.upper_level = self.HDI()
        self.interval = self.upper_level - self.lower_level
        self.map = self.MAP()
        
    def HDI(self):
        cdfvals = cumtrapz(self.posterior, self.param)
        sel = cdfvals > 0.
        x = self.param[1:][sel]
        cdfvals = cdfvals[sel]
        ppf = interp1d(cdfvals, x, fill_value=0., bounds_error=False)

        def intervalWidth(lowTailPr):
            ret = ppf(self.level + lowTailPr) - ppf(lowTailPr)
            if (ret > 0.):
                return ret
            else:
                return 1e4
        HDI_lowTailPr = fmin(intervalWidth, 1. - self.level, disp=self.verbose)[0]
        return ppf(HDI_lowTailPr), ppf(HDI_lowTailPr + self.level)


    def MAP(self):
        sp = UnivariateSpline(self.param, self.posterior, s=0.)
        x_highres = np.linspace(self.param[0], self.param[-1], 100000)
        y_highres = sp(x_highres)
        return x_highres[np.argmax(y_highres)]
        
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

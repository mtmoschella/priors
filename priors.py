class Prior:
    # abstract functor class used to supplement lnprior function for emcee
    __metaclass__ = ABCMeta # sets class to be abstract
    
    @abstractmethod
    def __call__(self, x):
        # any implementation must implement this function
        # returns the lnprior
        pass

    @abstractmethod
    def support(self):
        # any implementation must implement this function
        # returns the range of inputs such that the prior does not evaluate to -np.inf
        pass

class FlatPrior(Prior):
    def __init__(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
    
    def __call__(self, x):
        if (self.xmin<x<self.xmax):
            return 0.0
        else:
            return -np.inf
    
    def support(self):
        return (self.xmin, self.xmax)
        
class GaussianPrior(Prior):
    def __init__(self, xmean, xstd):
        self.xmean = xmean
        self.xstd = xstd
    
    def __call__(self, x):
        return -0.5*(x-xmean)**2/xstd**2

    def support(self):
        return (-np.inf, np.inf)

class LogNormalPrior(Prior):
    def __init__(self, xmean, xstd):
        # xmean, xstd are the mean & stddev of the lognormal distribution
        # mu, sigma are the mean & stddev of log(x), the gaussian parameters
        self.mu = np.log(xmean/np.sqrt(1+xstd**2/xmean**2))
        self.sigma = np.sqrt(np.log(1+xstd**2/xmean**2))

    def __call__(self, x):
        return -0.5*(np.log(x)-self.mu)**2/self.sigma**2 - np.log(x)

    def support(self):
        return (0.0, np.inf)


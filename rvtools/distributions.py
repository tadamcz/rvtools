import numpy as np
import scipy.stats


class Certainty(scipy.stats.rv_continuous):
    """
    The Dirac delta distribution, but at ``value`` instead of 0.

    Represents certainty, but as a continuous distribution.
    """

    def _argcheck(self, value):
        return np.isreal(value)

    def _pdf(self, x, value):
        return np.where(x == value, np.inf, 0)

    def _cdf(self, x, value):
        return np.where(x < value, 0, 1)

    def _ppf(self, x, value):
        return value

    def _rvs(self, value, size=None, random_state=None):
        return np.full(size, value)


certainty = Certainty()

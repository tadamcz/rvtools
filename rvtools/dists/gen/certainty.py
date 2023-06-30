import numpy as np
import scipy.stats


class Certainty(scipy.stats.rv_continuous):
    """
    Represents certainty, but as a continuous distribution, i.e. a subclass of
    ``scipy.stats.rv_continuous``.

    Like the Dirac delta distribution, but at ``value`` instead of 0.

    :param value: The value where all the probability density is located.

    Examples
    --------
    >>> import scipy
    >>> from rvtools.dists import certainty
    >>> dist = certainty(42)
    >>> dist.rvs(3)
    array([42, 42, 42])

    Has ``pdf`` and all other methods of a continuous distribution:

    >>> dist.pdf(0)
    0.0
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


# These being instances, not a classes, is not IMO idiomatic Python, but it's core to the way SciPy's
# ``rv_continuous`` class works. See examples of how SciPy defines their distributions in
# ``scipy/stats/_continuous_distns.py``.
certainty = Certainty()

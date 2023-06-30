import numpy as np
import scipy

from rvtools.dists.gen.tp_uniform import (
    argcheck,
    get_support,
    rvs,
    pdf_single,
    cdf_single,
    ppf_single,
    stats,
)


class HalvesUniform(scipy.stats.rv_continuous):
    """
    A piecewise uniform distribution with two pieces each having 0.5 probability mass.

    :param mini: The left bound of the distribution.
    :param sep: The boundary between the left and right pieces.
    :param maxi: The right bound of the distribution.

    Examples
    --------
    >>> from rvtools.dists import halves_uniform
    >>> dist = halves_uniform(0, 3, 10)
    """

    def _argcheck(self, mini, sep, maxi):
        return argcheck(mini, sep, maxi)

    def _get_support(self, mini, sep, maxi):
        return get_support(mini, sep, maxi)

    def _rvs(self, mini, sep, maxi, size=None, random_state=None):
        return rvs(mini, sep, maxi, size=size, random_state=random_state)

    def _pdf(self, x, mini, sep, maxi):
        return np.vectorize(pdf_single)(x, mini, sep, maxi)

    def _cdf(self, x, mini, sep, maxi):
        return np.vectorize(cdf_single)(x, mini, sep, maxi)

    def _ppf(self, p, mini, sep, maxi):
        return np.vectorize(ppf_single)(p, mini, sep, maxi)

    def _stats(self, mini, sep, maxi):
        return stats(mini, sep, maxi)


halves_uniform = HalvesUniform()

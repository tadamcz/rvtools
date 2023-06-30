from scipy.stats.distributions import rv_frozen

import rvtools.dists


def tp_uniform(mini, mode, maxi, psep=0.5) -> rv_frozen:
    """
    Create a (frozen) two-piece uniform distribution.

    Since SciPy does not support defining optional shape parameters, :py:obj:`rvtools.dists.tp_uniform`
    and :py:obj:`rvtools.dists.halves_uniform` must have two different signatures.

    This is a simple wrapper function that fixes this by accepting ``psep`` as an optional
    argument with a default of 0.5, which corresponds to a ``halves_uniform`` distribution.

    Examples
    --------

    >>> from rvtools.construct import tp_uniform
    >>> d1 = tp_uniform(0, 3, 12)
    >>> d2 = tp_uniform(0, 3, 12, psep=0.5)
    >>> d1.ppf(0.9)  == d2.ppf(0.9)
    True

    """
    return rvtools.dists.tp_uniform(mini, mode, maxi, psep=psep)

from betapert import mpert
from scipy.stats.distributions import rv_frozen


def pert(mini, mode, maxi, lambd=4) -> rv_frozen:
    """
    Create a (frozen) PERT or modified PERT distribution.

    Since SciPy does not support defining optional shape parameters, :py:obj:`rvtools.dists.pert`
    and :py:obj:`rvtools.dists.mpert` must have two different signatures.

    This is a simple wrapper function that fixes this by accepting ``lambd`` as an optional
    argument with a default of 4, which corresponds to the original PERT distribution.


    Examples
    --------

    >>> from rvtools.construct import pert
    >>> d1 = pert(0, 3, 12)
    >>> d2 = pert(0, 3, 12, lambd=4)
    >>> d1.ppf(0.9)  == d2.ppf(0.9)
    True

    """
    return mpert(mini, mode, maxi, lambd=lambd)

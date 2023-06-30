"""
Arbitrary parameters in SciPy's ``rv_continuous`` class must be 'shape' parameters. Optional shape parameters are
not supported, and are seemingly impossible to implement without egregious hacks. So there are two classes,
one for the halves-uniform distribution (with ``p=0.5``) and one for the more general two-piece uniform. Beyond being
repetitious, this also adversely affects the user-facing API.
"""

import numpy as np
import scipy


class TwoPieceUniform(scipy.stats.rv_continuous):
    """
    A piecewise uniform distribution with two pieces.

    :param mini: The left bound of the distribution.
    :param sep: The boundary between the left and right pieces.
    :param maxi: The right bound of the distribution.
    :param psep: The probability at ``sep``, i.e. the probability of the left piece, i.e. ``P(X < sep) = psep``.

    Examples
    --------
    >>> from rvtools.dists import tp_uniform
    >>> dist = tp_uniform(0, 3, 10, psep=0.1)
    """

    def _argcheck(self, mini, sep, maxi, psep):
        return argcheck(mini, sep, maxi, psep)

    def _get_support(self, mini, sep, maxi, psep):
        return get_support(mini, sep, maxi, psep)

    def _rvs(self, mini, sep, maxi, psep, size=None, random_state=None):
        return rvs(mini, sep, maxi, psep, size=size, random_state=random_state)

    def _pdf(self, x, mini, sep, maxi, psep):
        return np.vectorize(pdf_single)(x, mini, sep, maxi, psep)

    def _cdf(self, x, mini, sep, maxi, psep):
        return np.vectorize(cdf_single)(x, mini, sep, maxi, psep)

    def _ppf(self, p, mini, sep, maxi, psep):
        return np.vectorize(ppf_single)(p, mini, sep, maxi, psep)

    def _stats(self, mini, sep, maxi, psep):
        return stats(mini, sep, maxi, psep)


def argcheck(mini, sep, maxi, psep=0.5):
    return mini <= sep <= maxi and 0 <= psep <= 1


def get_support(mini, sep, maxi, psep=0.5):
    return mini, maxi


def rvs(mini, sep, maxi, psep=0.5, size=None, random_state=None):
    """
    With size proportional to p, sample from a uniform distribution on [mini, q]. With
    size propotional to 1-p, sample from a uniform distribution on [q, maxi].
    """
    if size is not None:
        if len(size) == 1:  # Caller will give this as (size,)
            size = size[0]
        else:
            raise NotImplementedError("size must be a scalar for TwoPieceUniform")
    size_left = np.ceil(size * psep).astype(int)
    size_right = size - size_left

    samples_left = np.random.uniform(mini, sep, size_left)
    samples_right = np.random.uniform(sep, maxi, size_right)
    return np.concatenate([samples_left, samples_right])


def pdf_single(x, mini, sep, maxi, psep=0.5):
    if mini <= x <= sep:
        return psep * _uniform_pdf(x, mini, sep)
    elif sep <= x <= maxi:
        return (1 - psep) * _uniform_pdf(x, sep, maxi)
    else:
        return 0


def cdf_single(x, mini, sep, maxi, psep=0.5):
    if x < mini:
        return 0
    elif mini <= x < sep:
        return psep * ((x - mini) / (sep - mini))
    elif sep <= x < maxi:
        return psep + (1 - psep) * ((x - sep) / (maxi - sep))
    else:  # x >= maxi
        return 1


def ppf_single(p, mini, sep, maxi, psep=0.5):
    if 0 <= p < psep:
        return mini + p / psep * (sep - mini)
    elif psep <= p <= 1:
        return sep + (p - psep) / (1 - psep) * (maxi - sep)
    else:
        return np.nan


def stats(mini, sep, maxi, psep=0.5):
    mean_left = (mini + sep) / 2
    mean_right = (sep + maxi) / 2
    mean = psep * mean_left + (1 - psep) * mean_right

    var = _var(mini, sep, maxi, psep)

    # These will be calculated (inefficiently) by SciPy's generic methods
    skew = None
    kurt = None

    return mean, var, skew, kurt


def _var(mini, sep, maxi, psep):
    # Calculate variance within each piece
    var1 = psep * (sep - mini) ** 2 / 12
    var2 = (1 - psep) * (maxi - sep) ** 2 / 12

    # Calculate the means of each piece
    mean1 = (mini + sep) / 2
    mean2 = (maxi + sep) / 2

    # Calculate the overall mean of the piecewise distribution
    overall_mean = psep * mean1 + (1 - psep) * mean2

    # Calculate the additional contribution to the due to the difference between the mean of each piece and the
    # overall mean. Each piece's mean contributes to the total variance in proportion to the square of its distance
    # from the overall mean, and the contribution is also weighted by the probability of the piece (since this is how
    # often we'd expect to be in this piece).
    var_mean_diff1 = psep * (mean1 - overall_mean) ** 2
    var_mean_diff2 = (1 - psep) * (mean2 - overall_mean) ** 2

    # Total variance
    variance = var1 + var2 + var_mean_diff1 + var_mean_diff2

    return variance


def _uniform_pdf(x, mini, maxi):
    """
    Helper. Using `scipy.stats.uniform`, would require converting to (loc, scale) form and make the code more confusing.
    """
    if mini <= x <= maxi:
        return 1 / (maxi - mini)
    else:
        return 0


# These being instances, not a classes, is not IMO idiomatic Python, but it's core to the way SciPy's
# ``rv_continuous`` class works. See examples of how SciPy defines their distributions in
# ``scipy/stats/_continuous_distns.py``.
tp_uniform = TwoPieceUniform()

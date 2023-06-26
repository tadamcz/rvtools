"""
Arbitrary parameters in SciPy's ``rv_continuous`` class must be 'shape' parameters. Optional shape parameters are
not supported, and are seemingly impossible to implement without egregious hacks. So there are two classes,
one for the halves-uniform distribution (with ``p=0.5``) and one for the more general two-piece uniform. Beyond being
repetitious, this also adversely affects the user-facing API.
"""

import numpy as np
import scipy


class TwoPieceUniform(scipy.stats.rv_continuous):
    def _argcheck(self, mini, q, maxi, p):
        return argcheck(mini, q, maxi, p)

    def _get_support(self, mini, q, maxi, p):
        return get_support(mini, q, maxi, p)

    def _rvs(self, mini, q, maxi, p, size=None, random_state=None):
        return rvs(mini, q, maxi, p, size, random_state)

    def _pdf(self, x, mini, q, maxi, p):
        return np.vectorize(pdf_single)(x, mini, q, maxi, p)

    def _cdf(self, x, mini, q, maxi, p):
        return np.vectorize(cdf_single)(x, mini, q, maxi, p)

    def _ppf(self, p_caller, mini, q, maxi, p):
        return np.vectorize(ppf_single)(p_caller, mini, q, maxi, p)

    def _stats(self, mini, q, maxi, p):
        return stats(mini, q, maxi, p)


class HalvesUniform(scipy.stats.rv_continuous):
    def _argcheck(self, mini, q, maxi):
        return argcheck(mini, q, maxi)

    def _get_support(self, mini, q, maxi):
        return get_support(mini, q, maxi)

    def _rvs(self, mini, q, maxi, size=None, random_state=None):
        return rvs(mini, q, maxi, size, random_state)

    def _pdf(self, x, mini, q, maxi):
        return np.vectorize(pdf_single)(x, mini, q, maxi)

    def _cdf(self, x, mini, q, maxi):
        return np.vectorize(cdf_single)(x, mini, q, maxi)

    def _ppf(self, p_caller, mini, q, maxi):
        return np.vectorize(ppf_single)(p_caller, mini, q, maxi)

    def _stats(self, mini, q, maxi):
        return stats(mini, q, maxi)


# ``tp_uniform`` and ``halves_uniform`` being instances, not a classes, is not IMO idiomatic Python, but it's core to
# the way SciPy's ``rv_continuous`` class works. See examples of how SciPy defines their distributions in
# ``scipy/stats/_continuous_distns.py``.
tp_uniform = TwoPieceUniform()
halves_uniform = HalvesUniform()


def argcheck(mini, q, maxi, p=0.5):
    return mini <= q <= maxi and 0 <= p <= 1


def get_support(mini, q, maxi, p=0.5):
    return mini, maxi


def rvs(mini, q, maxi, p=0.5, size=None, random_state=None):
    """
    With size proportional to p, sample from a uniform distribution on [mini, q]. With
    size propotional to 1-p, sample from a uniform distribution on [q, maxi].
    """
    size_left = np.ceil(size * p).astype(int)
    size_right = size - size_left

    samples_left = np.random.uniform(mini, q, size_left)
    samples_right = np.random.uniform(q, maxi, size_right)
    return np.concatenate([samples_left, samples_right])


def pdf_single(x, mini, q, maxi, p=0.5):
    if mini <= x <= q:
        return p * _uniform_pdf(x, mini, q)
    elif q <= x <= maxi:
        return (1 - p) * _uniform_pdf(x, q, maxi)
    else:
        return 0


def cdf_single(x, mini, q, maxi, p=0.5):
    if x < mini:
        return 0
    elif mini <= x < q:
        return p * ((x - mini) / (q - mini))
    elif q <= x < maxi:
        return p + (1 - p) * ((x - q) / (maxi - q))
    else:  # x >= maxi
        return 1


def ppf_single(p_caller, mini, q, maxi, p=0.5):
    if 0 <= p_caller < p:
        return mini + p_caller / p * (q - mini)
    elif p <= p_caller <= 1:
        return q + (p_caller - p) / (1 - p) * (maxi - q)
    else:
        return np.nan


def stats(mini, q, maxi, p=0.5):
    mean_left = (mini + q) / 2
    mean_right = (q + maxi) / 2
    mean = p * mean_left + (1 - p) * mean_right

    var = _var(mini, q, maxi, p)

    # These will be calculated (inefficiently) by SciPy's generic methods
    skew = None
    kurt = None

    return mean, var, skew, kurt


def _var(mini, q, maxi, p):
    # Calculate variance within each piece
    var1 = p * (q - mini) ** 2 / 12
    var2 = (1 - p) * (maxi - q) ** 2 / 12

    # Calculate the means of each piece
    mean1 = (mini + q) / 2
    mean2 = (maxi + q) / 2

    # Calculate the overall mean of the piecewise distribution
    overall_mean = p * mean1 + (1 - p) * mean2

    # Calculate the additional contribution to the due to the difference between the mean of each piece and the
    # overall mean. Each piece's mean contributes to the total variance in proportion to the square of its distance
    # from the overall mean, and the contribution is also weighted by the probability of the piece (since this is how
    # often we'd expect to be in this piece).
    var_mean_diff1 = p * (mean1 - overall_mean) ** 2
    var_mean_diff2 = (1 - p) * (mean2 - overall_mean) ** 2

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

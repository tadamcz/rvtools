import betapert
import scipy

from rvtools.dists.gen.certainty import Certainty


def is_frozen_norm(obj):
    """Returns ``True`` if and only if ``obj`` is a frozen normal distribution."""
    try:
        dist = obj.dist
    except AttributeError:
        return False
    return isinstance(dist, scipy.stats._continuous_distns.norm_gen)


def is_frozen_lognorm(obj):
    """Returns ``True`` if and only if ``obj`` is a frozen lognormal distribution."""
    try:
        dist = obj.dist
    except AttributeError:
        return False
    return isinstance(dist, scipy.stats._continuous_distns.lognorm_gen)


def is_frozen_beta(obj):
    """Returns ``True`` if and only if ``obj`` is a frozen beta distribution."""
    try:
        dist = obj.dist
    except AttributeError:
        return False
    return isinstance(dist, scipy.stats._continuous_distns.beta_gen)


def is_frozen_bernoulli(obj):
    """
    Returns ``True`` if and only if ``obj`` is a frozen Bernoulli distribution.
    """
    try:
        dist = obj.dist
    except AttributeError:
        return False
    return isinstance(dist, scipy.stats._discrete_distns.bernoulli_gen)


def is_frozen_certainty(obj):
    """Returns ``True`` if and only if ``obj`` is a frozen certainty distribution."""
    try:
        dist = obj.dist
    except AttributeError:
        return False
    return isinstance(dist, Certainty)


def is_frozen_pert(obj):
    """Returns ``True`` if and only if ``obj`` is a frozen PERT distribution."""
    try:
        dist = obj.dist
    except AttributeError:
        return False
    return isinstance(dist, betapert.PERT)

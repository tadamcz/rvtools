import betapert
import scipy

from rvtools.dists.gen.certainty import Certainty


def is_frozen_norm(obj):
    """Returns ``True`` if and only if ``obj`` is a frozen normal distribution."""
    try:
        return isinstance(obj.dist, scipy.stats._continuous_distns.norm_gen)
    except AttributeError:
        return False


def is_frozen_lognorm(obj):
    """Returns ``True`` if and only if ``obj`` is a frozen lognormal distribution."""
    try:
        return isinstance(obj.dist, scipy.stats._continuous_distns.lognorm_gen)
    except AttributeError:
        return False


def is_frozen_beta(obj):
    """Returns ``True`` if and only if ``obj`` is a frozen beta distribution."""
    try:
        return isinstance(obj.dist, scipy.stats._continuous_distns.beta_gen)
    except AttributeError:
        return False


def is_frozen_certainty(obj):
    """Returns ``True`` if and only if ``obj`` is a frozen certainty distribution."""
    try:
        return isinstance(obj.dist, Certainty)
    except AttributeError:
        return False


def is_frozen_pert(obj):
    """Returns ``True`` if and only if ``obj`` is a frozen PERT distribution."""
    try:
        return isinstance(obj.dist, betapert.PERT)
    except AttributeError:
        return False

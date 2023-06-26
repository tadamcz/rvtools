import scipy


def is_frozen_norm(distribution):
    try:
        return isinstance(distribution.dist, scipy.stats._continuous_distns.norm_gen)
    except AttributeError:
        return False


def is_frozen_lognorm(distribution):
    try:
        return isinstance(distribution.dist, scipy.stats._continuous_distns.lognorm_gen)
    except AttributeError:
        return False


def is_frozen_beta(distribution):
    try:
        return isinstance(distribution.dist, scipy.stats._continuous_distns.beta_gen)
    except AttributeError:
        return False

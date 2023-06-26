from numbers import Real

import numpy as np
import scipy

from rvtools.construct._helpers import parse_spec
from rvtools.construct.norm import params_from_quantiles as norm_params_from_quantiles


def lognorm(
    mu: Real = None,
    sigma: Real = None,
    *,
    mean: Real = None,
    sd: Real = None,
    quantiles: dict[Real, Real] = None,
    **kwargs,
):
    spec = parse_spec(mu=mu, sigma=sigma, mean=mean, sd=sd, quantiles=quantiles, **kwargs)
    if spec.keys() == {"mu", "sigma"}:
        return from_params(mu, sigma)
    elif spec.keys() == {"mean", "sd"}:
        return from_mean_sd(mean, sd)
    elif spec.keys() == {"quantiles"}:
        return from_quantiles(spec["quantiles"])
    else:
        raise ValueError(
            "You must specify either 'mu' and 'sigma', 'mean' and 'sd', or 'quantiles'."
        )


def from_quantiles(quantiles: dict[Real, Real]):
    if len(quantiles) != 2:
        raise ValueError(f"Expected exactly two quantiles, got {len(quantiles)}.")
    # Get the values of the quantiles
    ps = list(quantiles.keys())
    qs = list(quantiles.values())

    # The log of our random variable is normally distributed with mean mu and standard deviation sigma
    log_qs = np.log(qs)
    mu, sigma = norm_params_from_quantiles(ps[0], log_qs[0], ps[1], log_qs[1])

    return from_params(mu, sigma)


def from_params(mu: Real, sigma: Real):
    """
    SciPy's ``lognorm`` does not take the ``mu`` and ``sigma`` parameters (it takes its own
    ``scale`` and ``s`` parameters).

    This is a convenience wrapper that allows you to create a (frozen) SciPy log-normal distribution using ``mu`` and
    ``sigma``.
    """
    return scipy.stats.lognorm(scale=np.exp(mu), s=sigma)


def from_mean_sd(mean, sd):
    mu, sigma = to_mu_sigma(mean, sd)
    return from_params(mu, sigma)


def to_mu_sigma(mean, sd):
    """
    Get the ``mu`` and ``sigma`` parameters for a log-normal distribution with the given ``mean`` and ``sd``.
    """
    var = sd**2
    sigma = np.sqrt(np.log(var / (mean**2) + 1))

    mu = np.log(mean) - (1 / 2) * np.log(var / (mean**2) + 1)

    return mu, sigma

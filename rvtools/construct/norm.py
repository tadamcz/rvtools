from numbers import Real

import scipy

from rvtools.construct._helpers import parse_spec


def norm(mean: float = None, sd: float = None, *, quantiles: dict[Real, Real] = None, **kwargs):
    spec = parse_spec(mean=mean, sd=sd, quantiles=quantiles, **kwargs)
    if spec.keys() == {"mean", "sd"}:
        return scipy.stats.norm(mean, sd)
    elif spec.keys() == {"quantiles"}:
        return from_quantiles(spec["quantiles"])
    else:
        raise ValueError("You must specify either 'mean' and 'sd', or 'quantiles'.")


def from_quantiles(quantiles: dict[Real, Real]):
    if len(quantiles) != 2:
        raise ValueError(f"Expected exactly two quantiles, got {len(quantiles)}.")
    # Get the values of the quantiles
    ps = list(quantiles.keys())
    qs = list(quantiles.values())

    mu, sigma = params_from_quantiles(ps[0], qs[0], ps[1], qs[1])

    return scipy.stats.norm(mu, sigma)


def params_from_quantiles(p1, x1, p2, x2) -> tuple[Real, Real]:
    """Find parameters for a normal random variable X so that P(X < x1) = p1 and P(X < x2) = p2."""
    denom = scipy.stats.norm.ppf(p2) - scipy.stats.norm.ppf(p1)
    sigma = (x2 - x1) / denom
    mu = (x1 * scipy.stats.norm.ppf(p2) - x2 * scipy.stats.norm.ppf(p1)) / denom
    return mu, sigma

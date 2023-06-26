from numbers import Real

import scipy
from scipy.interpolate import interp1d

from rvtools.construct._helpers import parse_spec


def uniform(a: Real = None, b: Real = None, *, quantiles: dict[Real, Real] = None, **kwargs):
    spec = parse_spec(a=a, b=b, quantiles=quantiles, **kwargs)
    if spec.keys() == {"a", "b"}:
        return from_extrema(a, b)
    elif spec.keys() == {"quantiles"}:
        return from_quantiles(spec["quantiles"])
    else:
        raise ValueError("You must specify either the extrema 'a' and 'b', or 'quantiles'.")


def from_quantiles(quantiles: dict[Real, Real]):
    if len(quantiles) != 2:
        raise ValueError(f"Expected exactly two quantiles, got {len(quantiles)}.")
    # Get the values of the quantiles
    ps = list(quantiles.keys())
    qs = list(quantiles.values())

    # Calculate the minimum and maximum values
    f = interp1d(ps, qs, kind="linear", fill_value="extrapolate", assume_sorted=True)
    min_val = f(0)
    max_val = f(1)

    return uniform(min_val, max_val)


def from_extrema(a, b):
    """
    SciPy's ``uniform`` does not take the extrema of the distribution (it takes its own ``loc`` and ``scale``).

    This is a convenience wrapper that allows you to create a (frozen) SciPy uniform distribution using the extrema
    ``a`` and ``b``. ``a`` need not be less than ``b``.
    """
    min, max = sorted([a, b])
    return scipy.stats.uniform(loc=min, scale=max - min)

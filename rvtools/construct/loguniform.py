from numbers import Real

import numpy as np
import scipy
from scipy.interpolate import interp1d

from rvtools.construct._helpers import parse_spec


def loguniform(a: Real = None, b: Real = None, *, quantiles: dict[Real, Real] = None, **kwargs):
    """
    Create a (frozen) SciPy log-uniform distribution.

    You can specify the parameters in one of two ways.

    1. Using the bounds ``a`` and ``b`` of the distribution:

    >>> from rvtools.construct import loguniform
    >>> loguniform(1, 2)  # doctest: +ELLIPSIS
    <scipy.stats._distn_infrastructure.rv_continuous_frozen object at 0x...>

    2. Using quantiles:

    >>> loguniform(p5=0.1, p95=0.9) # doctest: +ELLIPSIS
    <scipy.stats._distn_infrastructure.rv_continuous_frozen object at 0x...>
    >>> loguniform(quantiles={1/1000: 0.1, 999/1000: 0.9})  # doctest: +ELLIPSIS
    <scipy.stats._distn_infrastructure.rv_continuous_frozen object at 0x...>
    """
    spec = parse_spec(a=a, b=b, quantiles=quantiles, **kwargs)
    if spec.keys() == {"a", "b"}:
        low, high = sorted([a, b])
        return scipy.stats.loguniform(low, high)
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

    # Convert to log scale
    log_qs = np.log(qs)

    # Calculate the minimum and maximum values of the corresponding uniform
    log_f = interp1d(ps, log_qs, kind="linear", fill_value="extrapolate", assume_sorted=True)
    log_min_val = log_f(0)
    log_max_val = log_f(1)

    min_val = np.exp(log_min_val)
    max_val = np.exp(log_max_val)

    return scipy.stats.loguniform(min_val, max_val)

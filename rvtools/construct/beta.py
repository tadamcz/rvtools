from numbers import Real

import numpy as np
import scipy

from rvtools.construct._helpers import parse_spec


def beta(alpha: Real = None, beta: Real = None, *, quantiles: dict[Real, Real] = None, **kwargs):
    """
    Create a (frozen) SciPy beta distribution.

    You can specify the parameters in one of two ways.

    1. Using ``alpha`` and ``beta``:

    >>> from rvtools.construct import beta
    >>> beta(1, 2)  # doctest: +ELLIPSIS
    <scipy.stats._distn_infrastructure.rv_continuous_frozen object at 0x...>

    2. Using quantiles:

    >>> beta(p5=0.1, p95=0.9) # doctest: +ELLIPSIS
    <scipy.stats._distn_infrastructure.rv_continuous_frozen object at 0x...>
    >>> beta(quantiles={1/1000: 0.1, 999/1000: 0.9})  # doctest: +ELLIPSIS
    <scipy.stats._distn_infrastructure.rv_continuous_frozen object at 0x...>

    """
    spec = parse_spec(alpha=alpha, beta=beta, quantiles=quantiles, **kwargs)
    if spec.keys() == {"alpha", "beta"}:
        return scipy.stats.beta(alpha, beta)
    elif spec.keys() == {"quantiles"}:
        return from_quantiles(spec["quantiles"])
    else:
        raise ValueError("You must specify either 'alpha' and 'beta', or 'quantiles'.")


def from_quantiles(quantiles: dict[Real, Real]):
    if len(quantiles) != 2:
        raise ValueError(f"Expected exactly two quantiles, got {len(quantiles)}.")
    # Get the values of the quantiles
    ps = list(quantiles.keys())
    qs = list(quantiles.values())

    alpha_init, beta_init = 1, 1
    fit = scipy.optimize.curve_fit(
        lambda x, alpha, beta: scipy.stats.beta.cdf(x, alpha, beta),
        xdata=qs,
        ydata=ps,
        p0=[alpha_init, beta_init],
    )

    # Since we estimated numerically, check that the estimated parameters give us the right quantiles
    alpha, beta = fit[0]
    fitted_ps = scipy.stats.beta.cdf(qs, alpha, beta)
    if not np.allclose(fitted_ps, ps):
        raise ValueError(
            f"Could not fit beta distribution to quantiles. "
            f"Expected probabilities {ps}, got fitted values {fitted_ps}."
        )

    return scipy.stats.beta(alpha, beta)

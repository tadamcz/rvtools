A collection of tools for conveniently working with SciPy probability distributions.

# Installation
```shell
pip install rvtools  # or `poetry add rvtools`
```
# [Documentation](https://tadamcz.com/rvtools/)

# Sample usage
```python
from rvtools.construct import lognorm, uniform, loguniform, CopulaJoint, beta
from rvtools.dists import pert, certainty
# PERT distribution: (min, mode, max) like triangular, but smoother shape
pert(1, 2, 4)

lognorm(mu=1, sigma=2) # SciPy equivalent: lognorm(scale=np.exp(mu), s=sigma). Hard to remember.
lognorm(mean=3, sd=4)  # Would need explicit calculation in SciPy
lognorm(p50=1, p95=2)  # Percentiles. Would need explicit calculation in SciPy

uniform(1, 2)  # SciPy equivalent: uniform(1, scale=2-1). Can trip you up.
uniform(2, 1)  # Not possible to invert the order in SciPy even though it's mathematically equivalent

loguniform(p10=42, p90=100)  # Percentiles. Would need explicit calculation in SciPy

# Arbitrary quantiles (not just integer percentiles)
lognorm(quantiles={0.0123: 1, 999/1000: 2})

# Copula
# Specify a joint probability distribution from arbitrary marginal distributions
# and pairwise rank correlations.
marginals = {
    "consumption elasticity": uniform(0.75, 3),
    "market return": lognorm(p50=0.05, p95=0.15),
    "risk of war": beta(2, 4),
}
tau = {
    # Missing pairs are assumed to be independent
    ("risk of war", "market return"): -0.5,
}
CopulaJoint(marginals, kendall_tau=tau)
```

# TODO
## Truncated lognormal
## Add https://github.com/tadamcz/metalogistic once quality is good enough
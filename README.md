A collection of tools for conveniently working with SciPy probability distributions.

# Installation
```shell
pip install rvtools  # or `poetry add rvtools`
```

# Usage
## Additional probability distributions
These are subclasses of `scipy.stats._distn_infrastructure.rv_generic` (the base class for all SciPy distributions). They behave like you'd expect from SciPy.

### Two-piece uniform

```python
from rvtools import tp_uniform, halves_uniform

# A piecewise uniform distribution with two pieces
# (0, 3) with probability mass 0.1, and (3, 10) with probability mass 0.9
tp_uniform(0, 3, 10, p=0.1)

# For convenience, there is also a distribution for the special case
# of each piece having 0.5 probability mass:
halves_uniform(0, 3, 10)
```

### Certainty

```python
import scipy
from rvtools import certainty
# Represents certainty as a continuous rather than discrete distribution.
# In other words, this is like the Dirac delta distribution, but at any
# value (e.g. 42), not just 0.
certainty(42)
assert isinstance(certainty, scipy.stats.rv_continuous)
```

### PERT
See [tadamcz/betapert](https://github.com/tadamcz/betapert), these functions are simply imported from there for convenience.
```python
from rvtools import pert, mpert
pert(1, 2, 4)
mpert(1, 2, 4, lambd=2.5)
```



## Constructors for common distributions

These are functions that return a 'frozen' distribution object (which in SciPy means a distribution with specific parameter values). This means `lognorm(1, 2).cdf(3)` can be used, but the `lognorm.cdf(3, 1, 2)` that you may be used to from SciPy will raise an `AttributeError` with rvtools. (Given how SciPy's distribution infrastructure works, improving this would require severe hacks, as far as I can tell.)

⚠️ **Many of these function signatures intentionally diverge from `scipy.stats`.** The behaviour should generally be clear from the names of the keyword arguments. However, if you blindly use _positional_ arguments and expect the same behaviour as `scipy.stats`, bad things will happen.

This is an opinionated API designed to be in line with common usage in mathematics and statistics, and my personal preferences. It may be a bad fit if you want to write code that is highly idiomatic in the Python scientific computing ecosystem.

### Lognormal

```python
from rvtools.construct import lognorm

# Three ways to specify the parameters:

# 1. Using (mu, sigma)
lognorm(mu=1, sigma=2)
# 2. Using (mean, sd)
lognorm(mean=3, sd=4)
# 3. Using quantiles, these two are equivalent:
lognorm(p50=1, p95=2)
lognorm(quantiles={0.5: 1, 0.95: 2})

# If two positional arguments are given, they are taken as mu, sigma. 
# (This differs from SciPy)
lognorm(1, 2)
```

### Beta

```python
from rvtools.construct import beta
# Two ways to specify parameters:

# 1. Using (alpha, beta). This is the same as SciPy.
beta(3, 4) 
# 2. Using quantiles
beta(p5=0.1, p95=0.9)
```

### Normal

```python
from rvtools.construct import norm
# Two ways to specify parameters:

# 1. Using (mean, sd). This is the same as SciPy.
norm(0, 1)
# 2. Using quantiles
norm(p12=0, p42=1)
```

### Uniform

```python
from rvtools.construct import uniform
# Two ways to specify parameters:

# 1. Using bounds. This differs from SciPy.
uniform(0, 1)
# Bounds do not need to be ordered. This is equivalent:
uniform(1, 0)

# 2. Quantiles
uniform(p5=0, p95=1)
uniform(quantiles={0.05: 0, 0.95: 1})
```
## 'Type' checkers
A 'frozen' distribution inherits from `scipy.stats._distn_infrastructure.rv_frozen`. This means its Python type does not expose the distribution family (e.g. whether it's a `norm` or `lognorm`). 

These functions let you check this at runtime (by looking at the `dist` attribute of a frozen distribution).

```python
from rvtools.types import is_frozen_norm, is_frozen_lognorm, is_frozen_beta
import scipy
import rvtools.construct

assert is_frozen_norm(scipy.stats.norm(0, 1))
assert is_frozen_norm(rvtools.construct.norm(p10=0, p90=1))
```

# TODO
## Truncated lognormal
## Add https://github.com/tadamcz/copula-wrapper once quality is good enough
## Add https://github.com/tadamcz/metalogistic once quality is good enough
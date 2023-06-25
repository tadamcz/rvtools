A collection of tools for conveniently working with SciPy probability distributions.

# Installation
```shell
pip install rvtools  # or `poetry add rvtools`
```

# Usage
## Distribution constructors

⚠️ **Many of these APIs intentionally diverge from `scipy.stats`.** The behaviour should generally be clear from the names of the keyword arguments. However, if you blindly use positional arguments and expect the same behavior as `scipy.stats`, bad things will happen.

This is an opinionated library. The API is designed to be in line with common usage in mathematics and statistics, and my personal preferences.
### Certainty

```python
from rvtools import certainty
certainty(42)
```

### PERT

```python
from rvtools import pert, mpert
pert(1, 2, 4)
mpert(1, 2, 4, lambd=2.5)
```

### Lognormal

```python
from rvtools import lognorm

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
from rvtools import beta
# Two ways to specify parameters:

# 1. Using (alpha, beta). This is the same as SciPy.
beta(3, 4) 
# 2. Using quantiles
beta(p5=0.1, p95=0.9)
```

### Normal

```python
from rvtools import norm
# Two ways to specify parameters:

# 1. Using (mean, sd). This is the same as SciPy.
norm(0, 1)
# 2. Using quantiles
norm(p12=0, p42=1)
```

### Uniform

```python
from rvtools import uniform
# Two ways to specify parameters:

# 1. Using bounds. This differs from SciPy.
uniform(0, 1)
# Bounds do not need to be ordered. This is equivalent:
uniform(1, 0)

# 2. Quantiles
uniform(p5=0, p95=1)
uniform(quantiles={0.05: 0, 0.95: 1})
```
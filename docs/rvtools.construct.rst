rvtools.construct
===================================
.. automodule:: rvtools.construct

.. autosummary::
    :toctree: generated
    :recursive:

    beta
    lognorm
    norm
    uniform
    loguniform
    CopulaJoint


These are constructors that return a 'frozen' distribution object (which in SciPy means a distribution with specific parameter values).

This means that for example the following works:

>>> from rvtools import lognorm
>>> lognorm(1, 2).cdf(3)
0.5196623384975168

but the following, which you may be used to from SciPy, will raise an ``AttributeError``:

>>> lognorm.cdf(3, 1, 2)
AttributeError: 'function' object has no attribute 'cdf'

Given how SciPy's distribution infrastructure works, improving this would require severe hacks, as far as I can tell.

**Many of these function signatures intentionally diverge** from ``scipy.stats``. The behaviour should generally be clear from the names of the keyword arguments. However, if you blindly use *positional* arguments and expect the same behaviour as ``scipy.stats``, bad things will sometimes happen.

This is an opinionated API designed to be in line with common usage in mathematics and statistics, and my personal preferences. It may be a bad fit if you want to write code that is highly idiomatic in the Python scientific computing ecosystem.

For arcane implementation reasons (see `tadamcz/copula-wrapper <https://github.com/tadamcz/copula-wrapper>`_) ``CopulaJoint`` is a class while the other constructors are functions. You shouldn't have to think about this: both return a frozen distribution object when called.
rvtools.dists
===================================

.. automodule:: rvtools.dists

.. autosummary::
    :toctree: generated
    :recursive:

    certainty
    tp_uniform
    halves_uniform
    pert
    mpert


All distributions in this subpackage inherit from ``scipy.stats.distributions.rv_continuous`` (the base class for all SciPy continuous distributions). They behave exactly like you'd expect from SciPy.

In particular, they can be used as both 'frozen' and 'unfrozen' distributions, whereas the constructors in ``rvtools.construct`` always return frozen distributions.

For example with ``halves_uniform``, this is the frozen usage:

>>> from rvtools.dists import halves_uniform
>>> dist = halves_uniform(0, 3, 10)
>>> dist.pdf(5)
0.07142857142857142


And this is the unfrozen usage:

>>> from rvtools.dists import halves_uniform
>>> halves_uniform.pdf(5, 0, 3, 10)
0.07142857142857142


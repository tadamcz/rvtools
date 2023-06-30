rvtools.fam
===================================

.. automodule:: rvtools.fam
    :members:


A 'frozen' distribution inherits from ``scipy.stats.distributions.rv_frozen``. This means its Python type does not expose the distribution family (e.g. whether it's a ``norm`` or ``lognorm``).

These functions let you check this at runtime (by looking at the ``dist`` attribute of a frozen distribution).

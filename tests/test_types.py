import scipy

import rvtools.types


def test_normal():
    assert rvtools.types.is_frozen_norm(scipy.stats.norm(1, 1))
    assert not rvtools.types.is_frozen_norm(scipy.stats.uniform(1, 1))


def test_lognormal():
    assert rvtools.types.is_frozen_lognorm(scipy.stats.lognorm(1, 1))
    assert not rvtools.types.is_frozen_lognorm(scipy.stats.uniform(1, 1))


def test_beta():
    assert rvtools.types.is_frozen_beta(scipy.stats.beta(1, 1))
    assert not rvtools.types.is_frozen_beta(scipy.stats.uniform(1, 1))

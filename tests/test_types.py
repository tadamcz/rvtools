import pytest
import scipy

import rvtools.types


@pytest.fixture
def frozen_wishart():
    return scipy.stats.wishart(1, 1)


def test_normal(frozen_wishart):
    assert rvtools.types.is_frozen_norm(scipy.stats.norm(1, 1))
    assert not rvtools.types.is_frozen_norm(frozen_wishart)


def test_lognormal(frozen_wishart):
    assert rvtools.types.is_frozen_lognorm(scipy.stats.lognorm(1, 1))
    assert not rvtools.types.is_frozen_lognorm(frozen_wishart)


def test_beta(frozen_wishart):
    assert rvtools.types.is_frozen_beta(scipy.stats.beta(1, 1))
    assert not rvtools.types.is_frozen_beta(frozen_wishart)


def test_certainty(frozen_wishart):
    assert rvtools.types.is_frozen_certainty(rvtools.certainty(1, 1))
    assert not rvtools.types.is_frozen_certainty(frozen_wishart)


def test_pert(frozen_wishart):
    assert rvtools.types.is_frozen_pert(rvtools.pert(1, 1, 1))
    assert not rvtools.types.is_frozen_pert(frozen_wishart)

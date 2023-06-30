import pytest
import scipy
import rvtools
from rvtools.fam import (
    is_frozen_norm,
    is_frozen_lognorm,
    is_frozen_beta,
    is_frozen_certainty,
    is_frozen_pert,
)


@pytest.fixture
def frozen_wishart():
    return scipy.stats.wishart(1, 1)


def test_normal(frozen_wishart):
    assert is_frozen_norm(scipy.stats.norm(1, 1))
    assert not is_frozen_norm(frozen_wishart)


def test_lognormal(frozen_wishart):
    assert is_frozen_lognorm(scipy.stats.lognorm(1, 1))
    assert not is_frozen_lognorm(frozen_wishart)


def test_beta(frozen_wishart):
    assert is_frozen_beta(scipy.stats.beta(1, 1))
    assert not is_frozen_beta(frozen_wishart)


def test_certainty(frozen_wishart):
    assert is_frozen_certainty(rvtools.dists.certainty(1, 1))
    assert not is_frozen_certainty(frozen_wishart)


def test_pert(frozen_wishart):
    assert is_frozen_pert(rvtools.dists.pert(1, 1, 1))
    assert not is_frozen_pert(frozen_wishart)

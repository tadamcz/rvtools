import pytest

from rvtools import certainty


@pytest.fixture
def value():
    return 0.123


@pytest.fixture
def dist(value):
    return certainty(value)


def test_cdf(dist, value):
    assert dist.cdf(value - 1e-10) == 0
    assert dist.cdf(value + 1e-10) == 1


def test_rvs(dist, value):
    assert all(dist.rvs(10) == [value] * 10)

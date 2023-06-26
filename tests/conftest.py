import numpy as np
import pytest


@pytest.fixture()
def random_seed():
    np.random.seed(2978654298)


def assert_same_distribution(d1, d2):
    """
    Assert that two (frozen) distributions are the same.
    """
    assert d1.support() == d2.support()

    xs = np.linspace(d1.ppf(0.001), d1.ppf(0.999))
    assert d1.pdf(xs) == pytest.approx(d2.pdf(xs))

    ps = np.linspace(0, 1)
    assert d1.ppf(ps) == pytest.approx(d2.ppf(ps))

    # ...We could check more things, but is there really any point?

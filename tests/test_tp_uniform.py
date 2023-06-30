import numpy as np
import pytest
import scipy

from rvtools.dists import tp_uniform
from rvtools.construct import uniform
from rvtools.dists import halves_uniform
from rvtools.dists.gen.tp_uniform import TwoPieceUniform
from tests.conftest import assert_same_distribution


@pytest.fixture(
    params=[
        # A simple case
        (0, 1, 3),
        # Negative and decimal values
        (-1, 0.1, 5),
        # q is very close to the minimum
        (0, 1 / 300, 1),
        # q is very close to the maximum
        (0, 299 / 300, 1),
        # Large values
        (1e6, 1e6 + 1, 1e6 + 2),
        # Small values
        (0, 1e-6, 2e-6),
    ],
    ids=lambda param: f"mini={param[0]}, q={param[1]}, maxi={param[2]}",
)
def param_triple(request):
    return request.param


@pytest.fixture(params=[0.90, 0.05, 1 / 3], ids=lambda p: f"p={p}")
def psep(request):
    return request.param


def test_rvs(param_triple, psep):
    mini, sep, maxi = param_triple
    dist = tp_uniform(mini=mini, sep=sep, maxi=maxi, psep=psep)

    # When size=1, deterministically only 1 sample from the left side is returned
    assert mini <= dist.rvs(size=1, random_state=0) <= sep


def test_pdf_any(param_triple, psep):
    mini, sep, maxi = param_triple
    dist = tp_uniform(mini=mini, sep=sep, maxi=maxi, psep=psep)

    assert dist.pdf(mini - 1) == pytest.approx(0)
    assert dist.pdf(maxi + 1) == pytest.approx(0)


def test_cdf_any(param_triple, psep):
    mini, sep, maxi = param_triple
    dist = tp_uniform(mini=mini, sep=sep, maxi=maxi, psep=psep)

    assert dist.cdf(mini - 1e-6) == pytest.approx(0)
    assert dist.cdf(sep) == pytest.approx(psep)
    assert dist.cdf(maxi + 1e-6) == pytest.approx(1)


def test_pdf_case_1():
    # Same width, pi=(0.5, 0.5)
    dist = tp_uniform(0, 1, 2, 0.5)
    assert dist.pdf(0.1) == pytest.approx(0.5)
    assert dist.pdf(1.1) == pytest.approx(0.5)


def test_pdf_case_2():
    # Same width, pi=(0.3, 0.7)
    dist = tp_uniform(0, 1, 2, 0.3)
    assert dist.pdf(0.1) == pytest.approx(0.3)
    assert dist.pdf(1.1) == pytest.approx(0.7)


def test_pdf_case_3():
    # pi=(0.5, 0.5), right is twice as wide as left
    dist = tp_uniform(0, 1, 3, 0.5)
    assert dist.pdf(0.1) == pytest.approx(0.5)
    assert dist.pdf(1.1) == pytest.approx(0.25)


def test_pdf_case_4():
    # pi=(1/3, 2/3), right is twice as wide as left but also has twice the probability mass
    dist = tp_uniform(0, 1, 3, 1 / 3)
    assert dist.pdf(0.1) == pytest.approx(dist.pdf(1.1))


def test_uniform_special_case(param_triple):
    mini, _, maxi = param_triple

    dist = tp_uniform(mini, mini, maxi, 0)
    assert_same_distribution(dist, uniform(mini, maxi))

    dist = tp_uniform(mini, maxi, maxi, 1)
    assert_same_distribution(dist, uniform(mini, maxi))


@pytest.fixture()
def numerical_cdf_dist(param_triple, psep):
    class NumericalCDFTwoPieceUniform(TwoPieceUniform):
        def _cdf(self, x, *args, **kwargs):
            return scipy.stats.rv_continuous._cdf(self, x, *args, **kwargs)

    # Follow SciPy pattern
    numerical_cdf_tp_uniform = NumericalCDFTwoPieceUniform()

    return numerical_cdf_tp_uniform(*param_triple, psep)


def test_cdf_matches_numerical(param_triple, psep, numerical_cdf_dist):
    """
    The CDF (which we have in closed form) matches the CDF obtained by numerically integrating the PDF.
    """
    mini, sep, maxi = param_triple
    x = np.linspace(mini, maxi, 10)

    numerical = numerical_cdf_dist.cdf(x)
    closed_form = tp_uniform.cdf(x, mini, sep, maxi, psep)

    assert numerical == pytest.approx(closed_form)


def test_mean_var(param_triple, psep, random_seed):
    """
    mean() and var() (closed form) matches empirical mean and variance from sampling

    Not using expect() here because the kinks in the distribution make it hard for numerical integration to be accurate.
    """
    mini, sep, maxi = param_triple
    dist = tp_uniform(mini, sep, maxi, psep)
    samples = dist.rvs(size=int(1e7))

    assert dist.mean() == pytest.approx(np.mean(samples), rel=1 / 1000)
    assert dist.var() == pytest.approx(np.var(samples), rel=1 / 1000)


def test_generalization(param_triple):
    mini, sep, maxi = param_triple
    assert_same_distribution(tp_uniform(mini, sep, maxi, 0.5), halves_uniform(mini, sep, maxi))


def test_rvs_does_not_raise(param_triple):
    mini, sep, maxi = param_triple
    dist = tp_uniform(mini, sep, maxi, 0.5)
    dist.rvs(size=10)

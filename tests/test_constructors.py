import numpy as np
import pytest
import scipy

import rvtools.construct as construct
from tests.conftest import assert_same_distribution


def assert_has_quantiles(d, quantiles):
    """
    Assert that a (frozen) distribution has the given quantiles.
    """
    for p, q in quantiles.items():
        assert d.cdf(q) == pytest.approx(p)


class TestBeta:
    @pytest.fixture(params=[0.5, 3], ids=lambda p: f"alpha={p}")
    def alpha(self, request):
        return request.param

    @pytest.fixture(params=[1, 5], ids=lambda p: f"beta={p}")
    def beta(self, request):
        return request.param

    @pytest.mark.parametrize("as_kwargs", [True, False], ids=lambda x: "kwargs" if x else "args")
    def test_from_alpha_beta(self, alpha, beta, as_kwargs):
        if as_kwargs:
            our_d = construct.beta(alpha=alpha, beta=beta)
        else:
            our_d = construct.beta(alpha, beta)
        scipy_d = scipy.stats.beta(alpha, beta)
        assert_same_distribution(our_d, scipy_d)

    @pytest.fixture(
        params=[{0.1: 0.2, 0.5: 0.85}, {0.1: 0.5, 0.5: 0.99}, {0.001: 0.5, 0.5: 0.99}],  #  #  #
        ids=lambda p: f"quantiles={p}",
    )
    def quantiles(self, request):
        return request.param

    def test_from_quantiles(self, quantiles):
        assert_has_quantiles(construct.beta(quantiles=quantiles), quantiles)

    def test_too_many_args(self):
        with pytest.raises(ValueError, match="You must specify"):
            construct.beta(1, 1, quantiles={0.1: 0.1, 0.5: 0.5})


class TestLognorm:
    @pytest.fixture(params=[0.5, 3], ids=lambda p: f"mean={p}")
    def mean(self, request):
        return request.param

    @pytest.fixture(params=[1, 5], ids=lambda p: f"sd={p}")
    def sd(self, request):
        return request.param

    @pytest.fixture(params=[0.5, 3], ids=lambda p: f"mu={p}")
    def mu(self, request):
        return request.param

    @pytest.fixture(params=[1, 5], ids=lambda p: f"sigma={p}")
    def sigma(self, request):
        return request.param

    @pytest.fixture
    def quantiles(self):
        return {0.1: 1, 0.9: 10}

    @pytest.mark.parametrize("as_kwargs", [True, False], ids=lambda x: "kwargs" if x else "args")
    def test_from_mu_sigma(self, mu, sigma, as_kwargs):
        if as_kwargs:
            dist = construct.lognorm(mu=mu, sigma=sigma)
        else:
            dist = construct.lognorm(mu, sigma)

        # Check that the log of our distribution is normal
        ps = np.linspace(0, 1)
        got = np.log(dist.ppf(ps))
        want = scipy.stats.norm.ppf(ps, mu, sigma)
        assert got == pytest.approx(want)

    def test_from_mean_sd(self, mean, sd):
        dist = construct.lognorm(mean=mean, sd=sd)
        got = (dist.mean(), dist.std())
        assert got == pytest.approx((mean, sd))

    def test_from_quantiles(self, quantiles):
        dist = construct.lognorm(quantiles=quantiles)
        assert_has_quantiles(dist, quantiles)

    def test_inconsistent_spec(self):
        with pytest.raises(ValueError, match="You must specify"):
            construct.lognorm(1, 1, quantiles={0.1: 0.1, 0.5: 0.5})
        with pytest.raises(ValueError, match="You must specify"):
            construct.lognorm(1, 1, mean=1)
        with pytest.raises(ValueError, match="You must specify"):
            construct.lognorm(mean=1, sd=1, mu=1)


class TestNorm:
    @pytest.fixture(params=[0.5, 3], ids=lambda p: f"mean={p}")
    def mean(self, request):
        return request.param

    @pytest.fixture(params=[1, 5], ids=lambda p: f"sd={p}")
    def sd(self, request):
        return request.param

    @pytest.fixture
    def quantiles(self):
        return {0.1: -1, 0.9: 1}

    def test_from_mean_sd(self, mean, sd):
        our_d = construct.norm(mean=mean, sd=sd)
        scipy_d = scipy.stats.norm(mean, sd)
        assert_same_distribution(our_d, scipy_d)

    def test_from_quantiles(self, quantiles):
        dist = construct.norm(quantiles=quantiles)
        assert_has_quantiles(dist, quantiles)

    def test_too_many_args(self):
        with pytest.raises(ValueError, match="You must specify"):
            construct.norm(1, 1, quantiles={0.1: 0.1, 0.5: 0.5})


class TestUniform:
    @pytest.fixture(
        params=[
            # A simple case
            (1, 2),
            # Need not be in order
            (2, 1),
            # Small numbers
            (1e-10, 1e-9),
        ]
    )
    def pair(self, request):
        return request.param

    @pytest.fixture(params=[{0.1: 1, 0.9: 2}], ids=lambda p: f"quantiles={p}")
    def quantiles(self, request):
        return request.param

    def test_from_pair(self, pair):
        dist = construct.uniform(*pair)
        assert dist.ppf(0.5) == pytest.approx(np.mean(pair))
        assert dist.support() == pytest.approx(sorted(pair))

    def test_from_quantiles(self, quantiles):
        dist = construct.uniform(quantiles=quantiles)
        assert_has_quantiles(dist, quantiles)

    def test_degenerate_rvs(self):
        our_d = construct.uniform(1, 1)
        assert all(our_d.rvs(size=10) == [1] * 10)

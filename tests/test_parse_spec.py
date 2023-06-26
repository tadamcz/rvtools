import pytest

from rvtools.construct._helpers import parse_spec, collect_kwarg_quantiles


class TestParseSpec:
    def test_no_quantiles(self):
        spec = parse_spec(a=1, b=2)
        assert spec == {"a": 1, "b": 2}

    def test_dict_quantiles(self):
        spec = parse_spec(quantiles={0.1: 10, 0.5: 50, 0.9: 90})
        assert spec == {"quantiles": {0.1: 10, 0.5: 50, 0.9: 90}}

    def test_kwarg_quantiles(self):
        spec = parse_spec(p10=10, p50=50, p90=90)
        assert spec == {"quantiles": {0.1: 10, 0.5: 50, 0.9: 90}}

    def test_both_types_quantiles(self):
        with pytest.raises(
            ValueError,
            match="Cannot specify quantiles both as a dictionary and as keyword arguments.",
        ):
            parse_spec(quantiles={0.1: 10, 0.5: 50, 0.9: 90}, p10=10, p50=50, p90=90)

    def test_invalid_dict_quantiles(self):
        with pytest.raises(ValueError, match="Must be a number between 0 and 1."):
            parse_spec(quantiles={-0.1: 10, 1.5: 50})


class TestCollectKwargQuantiles:
    def test_collect_kwarg_quantiles(self):
        kwargs = {"p10": 10, "p50": 50, "p90": 90}
        quantiles = collect_kwarg_quantiles(kwargs)
        assert quantiles == {0.1: 10, 0.5: 50, 0.9: 90}
        assert kwargs == {}

    def test_non_quantile_param_startswith_p(self):
        """
        Parameters starting with 'p' that are not quantiles should be treated correctly.
        """
        kwargs = {"p1aram": 10, "a": 5, "p10": 123}
        quantiles = collect_kwarg_quantiles(kwargs)
        assert quantiles == {0.1: 123}
        assert kwargs == {"p1aram": 10, "a": 5}

    def test_leading_zero_in_quantiles(self):
        """
        Quantiles with a leading zero should be interpreted correctly
        """
        kwargs = {"p05": 10, "p10": 20, "p90": 90}
        quantiles = collect_kwarg_quantiles(kwargs)
        assert quantiles == {0.05: 10, 0.1: 20, 0.9: 90}
        assert kwargs == {}

    def test_invalid_quantiles(self):
        with pytest.raises(ValueError, match="Invalid quantile name"):
            kwargs = {"p-10": 10, "p150": 50}
            collect_kwarg_quantiles(kwargs)

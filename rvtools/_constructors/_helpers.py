from enum import Enum


def build_spec(**kwargs):
    if "quantiles" in kwargs and kwargs["quantiles"] is not None:
        for k, v in kwargs["quantiles"].items():
            if not 0 <= k <= 1:
                raise ValueError(f"Quantiles must be between 0 and 1, got {k}.")
    return filter_none(**kwargs)


def filter_none(**kwargs):
    """
    Filter out the ``None`` values from the keyword arguments, and return the result as a dict.
    """
    return {k: v for k, v in kwargs.items() if v is not None}

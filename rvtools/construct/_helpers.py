def parse_spec(**kwargs):
    dict_quantiles = kwargs.get("quantiles")
    kwargs_quantiles = collect_kwarg_quantiles(kwargs)
    if dict_quantiles and kwargs_quantiles:
        raise ValueError(f"Cannot specify quantiles both as a dictionary and as keyword arguments.")
    if kwargs_quantiles:
        kwargs["quantiles"] = kwargs_quantiles

    if kwargs.get("quantiles"):
        for p in kwargs["quantiles"].keys():
            if not 0 <= p <= 1:
                raise ValueError(f"Invalid quantile: {p}. Must be a number between 0 and 1.")

    return filter_none(**kwargs)


def collect_kwarg_quantiles(kwargs):
    """
    Collect the quantiles from the keyword arguments, and return them as a dict. Delete the quantile keyword
    arguments in-place.
    """
    quantiles = {}
    names = list(kwargs.keys())
    for name in names:
        value = kwargs[name]
        if name.startswith("p"):
            try:
                p = int(name[1:])
            except ValueError:  # not a quantile argument
                continue
            if not 0 <= p <= 100:
                raise ValueError(
                    f"Invalid quantile name: {name}. Must be of the form 'pN' where N is a number between 0 and 100."
                )
            quantiles[p / 100] = value
            del kwargs[name]
    return quantiles


def filter_none(**kwargs):
    """
    Filter out the ``None`` values from the keyword arguments, and return the result as a dict.
    """
    return {k: v for k, v in kwargs.items() if v is not None}

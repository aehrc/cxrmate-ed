import torch


def compute_time_delta(event_time, reference_time, time_delta_map, denominator = 3600, to_tensor=True):
    """
    How to we transform time delta inputs? It appears that minutes are used as the input to 
    a weight matrix in "Self-Supervised Transformer for Sparse and Irregularly Sampled Multivariate 
    Clinical Time-Series". This is almost confirmed by the CVE class defined here:
    https://github.com/sindhura97/STraTS/blob/main/strats_notebook.ipynb, where the input has 
    a size of one.
    """
    time_delta = reference_time - event_time
    time_delta = time_delta.total_seconds() / (denominator)
    assert isinstance(time_delta, float), f'time_delta should be float, not {type(time_delta)}.'
    if time_delta < 0:
        raise ValueError(f'time_delta should be greater than or equal to zero, not {time_delta}.')
    time_delta = time_delta_map(time_delta)
    if to_tensor: 
        time_delta = torch.tensor(time_delta)
    return time_delta

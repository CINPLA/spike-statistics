import numpy as np


def nonstationary_poisson(times, rate):
    """
    Non-stationary Poisson process
    Implemented by Alexander Stasik (a.j.stasik@fys.uio.no)
    Parameters
    ----------
    times : array
        timepoints corresponding to rate.
    rate : array
        rates as function of time.
    Returns
    -------
    events : array
        time points from a poisson process which rate varies according to rate.
    Authors
    -------
    Alex Stasik, Mikkel Lepperød
    """
    n_exp = rate.max() * (times.max()-times.min())
    t_events = np.sort(
        np.random.uniform(
            times.min(), times.max(), np.random.poisson(n_exp)))
    mask = np.digitize(t_events, times)
    ratio = rate[mask] / rate.max()
    mask = np.random.uniform(0., 1., len(ratio)) < ratio
    return t_events[mask]


def stationary_poisson(t_start, t_stop, rate):
    """
    Stationary Poisson process
    Parameters
    ----------
    t_start : float
        Start time of the process (lower bound).
    t_stop : float
        Stop time of the process (upper bound).
    rate : float
        rate of the Poisson process
    Returns
    -------
    events : array
        time points from a Poisson process with rate rate.
    """
    n_exp = rate * (t_stop - t_start)
    return np.sort(
        np.random.uniform(
            t_start, t_stop, np.random.poisson(n_exp)))

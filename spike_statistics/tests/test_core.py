import pytest
import numpy as np
from spike_statistics.process import stationary_poisson, nonstationary_poisson
np.random.seed(12345)


def test_fano():
    #The fano factor is 1 for Poisson processes
    from spike_statistics.core import fano_factor
    t1 = [stationary_poisson(0.0, 1, 1) for _ in range(1000)]
    assert abs(fano_factor(t1) - 1) < .02


def test_nonstationary_poisson():
    times = np.arange(0, 100, .1)
    amp = 10
    f = .1
    rate = amp * (np.sin(2 * np.pi * f * times) + 1)
    ms = []
    for _ in range(100):
        n_p = nonstationary_poisson(times, rate)
        bins = np.arange(0, times.max(), 1)
        h,_ = np.histogram(n_p, bins)
        ms.append(np.mean(h))
    assert abs(np.mean(ms) - amp) < .1
